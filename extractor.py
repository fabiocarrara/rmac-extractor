import os
import cv2
import sys
import h5py
import time
import argparse
import tempfile
import numpy as np

import dask.array as da
from tqdm import tqdm

os.environ['GLOG_minloglevel'] = '2'
import caffe


class RMACExtractor:

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.means = np.array([103.93900299,  116.77899933,  123.68000031], dtype=np.float32)[None, :, None, None]
        
        proto = 'net/deploy_resnet101_normpython.prototxt'
        weights = 'net/model.caffemodel'
        if self.device_id < 0:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(self.device_id)
        
        self.net = caffe.Net(proto, caffe.TEST, weights=weights)
        self.net.forward(end='rmac/normalized')  # warm-start
        
        
    def pack_regions_for_network(self, all_regions):
        n_regs = np.sum([len(e) for e in all_regions])
        R = np.zeros((n_regs, 5), dtype=np.float32)
        cnt = 0
        # There should be a check of overflow...
        for i, r in enumerate(all_regions):
            try:
                R[cnt:cnt + r.shape[0], 0] = i
                R[cnt:cnt + r.shape[0], 1:] = r
                cnt += r.shape[0]
            except:
                continue
        assert cnt == n_regs
        R = R[:n_regs]
        # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
        R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
        R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1
        return R
        
        
    def get_rmac_region_coordinates(self, H, W, L):
        # Almost verbatim from Tolias et al Matlab implementation.
        # Could be heavily pythonized, but really not worth it...
        # Desired overlap of neighboring regions
        ovr = 0.4
        # Possible regions for the long dimension
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)

        b = (np.maximum(H, W) - w) / (steps - 1)
        # steps(idx) regions for long dimension. The +1 comes from Matlab
        # 1-indexing...
        idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1

        # Region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx
        elif H > W:
            Hd = idx

        regions_xywh = []
        for l in range(1, L+1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            # Center coordinates
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            # Center coordinates
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

            for i_ in cenH:
                for j_ in cenW:
                    regions_xywh.append([j_, i_, wl, wl])

        # Round the regions. Careful with the borders!
        for i in range(len(regions_xywh)):
            for j in range(4):
                regions_xywh[i][j] = int(round(regions_xywh[i][j]))
            if regions_xywh[i][0] + regions_xywh[i][2] > W:
                regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
            if regions_xywh[i][1] + regions_xywh[i][3] > H:
                regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
        return np.array(regions_xywh).astype(np.float32)
        

    def prepare_image(self, im, S):
        # Get aspect ratio and resize such as the largest side equals S
        im_size_hw = np.array(im.shape[0:2])
        ratio = float(S) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        I = cv2.resize(im, (new_size[1], new_size[0]))
        # Transpose for network and subtract mean
        I = I.transpose(2, 0, 1) - self.means
        return I


    def load_and_prepare_image(self, fname, S):
        im = cv2.imread(fname, 1)  # Read image always as 3-channel BGR
        I = self.prepare_image(im, S)
        return I
        
        
    def extract_from_urls(self, urls, out, S=550, L=2):
        print 'Extracting (S={}, L={}): {}'.format(S, L, out)
        
        features_db = None
        n_images = len(urls)
        
        for i, url in enumerate(tqdm(urls)):
            img = self.load_and_prepare_image(url, S)   
            features = self.extract_from_image(img, L)
            if features_db is None:
                features_db = h5py.File(out, 'w')
                features_dataset = features_db.create_dataset('rmac', (n_images, 2048), dtype=features.dtype)
                
            features_dataset[i] = features
            if i % 1000 == 0:
                features_db.flush()
                
        features_db.flush()
        return features_dataset

    def extract_from_pil(self, pil_img, S=550, L=2):
        pil_img = pil_img.convert('RGB')
        img = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR
        return self.extract_from_np(img, S=S, L=L)

    def extract_from_np(self, img, S=550, L=2):
        img = self.prepare_image(img, S)

        # print 'Reshaping net...'
        self.net.blobs['data'].reshape(*img.shape)
        # print 'Packing regions...'
        all_regions = []
        all_regions.append(self.get_rmac_region_coordinates(img.shape[2], img.shape[3], L))
        R = self.pack_regions_for_network(all_regions)
    
        self.net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        self.net.blobs['rois'].data[:] = R.astype(np.float32)
        # print 'Performing forward...'
        self.net.blobs['data'].data[:] = img
        self.net.forward(end='rmac/normalized')
        
        return np.squeeze(self.net.blobs['rmac/normalized'].data).copy()


    def extract_from_image(self, img, L):
        # print 'Reshaping net...'
        self.net.blobs['data'].reshape(*img.shape)
        # print 'Packing regions...'
        all_regions = []
        all_regions.append(self.get_rmac_region_coordinates(img.shape[2], img.shape[3], L))
        R = self.pack_regions_for_network(all_regions)
    
        self.net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        self.net.blobs['rois'].data[:] = R.astype(np.float32)
        # print 'Performing forward...'
        self.net.blobs['data'].data[:] = img
        self.net.forward(end='rmac/normalized')
        
        return np.squeeze(self.net.blobs['rmac/normalized'].data).copy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RMAC feature extractor')
    parser.add_argument('image_list', type=str, help='File containing the list of images URLs')
    parser.add_argument('output_db', type=str, help='Prefix of output HDF5 files')
    parser.add_argument('-s', '--sizes', nargs='+', type=int, default=[550, 800, 1050], help='Image longest side size. Accept multiple sizes (default: [550, 800, 1050])')
    parser.add_argument('-l', '--levels', type=int, default=2, help='Use L spatial levels (default: 2)')
    parser.add_argument('-d', '--device_id', type=int, default=-1, help='Device index of the GPU to use, -1 for CPU')
    parser.add_argument('-a', '--aggregate', action='store_true', help='Produce the aggregated multi-resolution features')
    args = parser.parse_args()
    
    assert os.path.exists(args.image_list), "List file not found."
    
    urls = [line.rstrip() for line in open(args.image_list, 'rb')]
    n_images = len(urls)
    print 'Found {} images.'.format(n_images)
    
    print 'Loading the extractor...'
    extractor = RMACExtractor(args.device_id)
    
    rmacs = [
        extractor.extract_from_urls(
            urls, '{}_S{}.h5'.format(args.output_db, size),
            S=size, L=args.levels)
        for size in args.sizes ]
    
    if args.aggregate:
        feat_dim = rmacs[0].shape[1]
        #tmp = [da.from_hdf5(f, chunk_size=(5000, feat_dim)) for f in rmacs]
        rmacs = da.stack(rmacs, axis=-1).sum(axis=2)
        rmacs /= da.sqrt((rmacs * rmacs).sum(axis=1))[:, None]
        aggr_out_db = '{}_S{}.h5'.format(args.output_db, '+'.join(map(str, args.sizes)))
        print 'Computing aggregated RMAC features:', aggr_out_db
        da.to_hdf5(aggr_out_db, '/rmac', rmacs)
