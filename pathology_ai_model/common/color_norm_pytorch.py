import numpy as np
import os
import cv2
import openslide
import pyvips
from PIL import Image
import torch

from .estimate_w import Wfast

Image.MAX_IMAGE_PIXELS = None


dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


class ColorNorm(object):
    def __init__(self, num_stains, lamb, img_level, _max=2000, background_correction=True):
        self.nstains = num_stains
        self.lamb = lamb
        self.img_level = img_level
        self._max = _max
        self.bg_correction = background_correction

    @staticmethod
    def numpy2vips(a):
        height, width, bands = a.shape
        linear = a.reshape(width * height * bands)
        vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                          dtype_to_format[str(a.dtype)])
        return vi

    def get_params_on_images(self, name, reference_FLAG=True):
        I = openslide.open_slide(name)
        if self.img_level >= I.level_count:
            print("Level", self.img_level, "unavailable for image, proceeding with level 0")
            level = 0
        else:
            level = self.img_level

        xdim, ydim = I.level_dimensions[level]
        ds = I.level_downsamples[level]

        print("Stain Separation in progress:", name, str(xdim) + str("x") + str(ydim))

        # parameters for W estimation
        num_patches = 20
        patchsize = 1000  # length of side of square

        i0_default = torch.tensor([255., 255., 255.]).type(torch.float32).cuda()

        if not self.bg_correction:
            print("Background correction disabled, default background intensity assumed")
            i0 = i0_default
        else:
            Wi, i0 = Wfast(I, self.nstains, self.lamb, num_patches, patchsize, level, self.bg_correction)

            assert Wi is not None, "Color Basis Matrix Estimation failed...image normalization skipped"
            Wi = torch.tensor(Wi).type(torch.float32).cuda()
            i0 = i0_default if i0 is None else torch.tensor(i0).type(torch.float32).cuda()

        print("Color Basis Matrix:", Wi)
        print("Image Background white intensity:", i0)

        if (xdim * ydim) <= (self._max * self._max):
            print("Small image processing...")
            img = np.asarray(I.read_region((0, 0), level, (xdim, ydim)), dtype=np.float32)[:, :, :3]
            img = torch.tensor(img).cuda()

            shape_tuple = img.shape
            Img_vecd = torch.reshape(torch.min(img, i0), [shape_tuple[0] * shape_tuple[1], shape_tuple[2]])
            V = torch.log(i0 + 1.0) - torch.log(Img_vecd + 1.0)
            Wi_inv = torch.transpose(torch.pinverse(Wi), dim0=0, dim1=1)
            Hiv = torch.relu(torch.matmul(V.type(torch.float32), Wi_inv))

            # Hta_Rmax = np.percentile(Hiv,q=99.,axis=0)
            H_Rmax = np.ones((self.nstains,), dtype=np.float32)
            for i in range(self.nstains):
                t = Hiv[:, i].cpu().numpy()
                H_Rmax[i] = np.percentile(t[t > 0], q=99., axis=0)


        else:
            _maxtf = 2550  # changed from initial 3000
            x_max = xdim
            y_max = min(max(int(_maxtf * _maxtf / x_max), 1), ydim)
            print("Large image processing...")
            self.Hiv_temp = np.memmap('H_target', dtype='float32', mode='w+', shape=(xdim * ydim, 2))

            x_tl = range(0, xdim, x_max)
            y_tl = range(0, ydim, y_max)
            print("WSI divided into", str(len(x_tl)) + "x" + str(len(y_tl)))
            count = 0
            print("Patch-wise H calculation in progress...")
            ind = 0
            perc = []
            for x in x_tl:
                for y in y_tl:
                    count += 1
                    xx = min(x_max, xdim - x)
                    yy = min(y_max, ydim - y)
                    print("Processing:", count, "		patch size", str(xx) + "x" + str(yy)),
                    img = np.asarray(I.read_region((int(ds * x), int(ds * y)), level, (xx, yy)), dtype=np.float32)[:, :, :3]
                    img = torch.tensor(img).cuda()
                    shape_tuple = img.shape

                    Img_vecd = torch.reshape(torch.min(img, i0), [shape_tuple[0] * shape_tuple[1], shape_tuple[2]])
                    V = torch.log(i0 + 1.0) - torch.log(Img_vecd + 1.0)
                    Wi_inv = torch.transpose(torch.pinverse(Wi), dim0=0, dim1=1)
                    Hiv = torch.relu(torch.matmul(V.type(torch.float32), Wi_inv))

                    self.Hiv_temp[ind:ind + len(Hiv), :] = Hiv.cpu().numpy()
                    _Hso_Rmax = np.ones((self.nstains,), dtype=np.float32)
                    for i in range(self.nstains):
                        t = Hiv[:, i].cpu().numpy()
                        _Hso_Rmax[i] = np.percentile(t[t > 0], q=99., axis=0)
                    perc.append([_Hso_Rmax[0], _Hso_Rmax[1]])
                    ind += len(Hiv)

            H_Rmax = np.percentile(np.array(perc), 50, axis=0).astype(np.float32)

        H_Rmax = torch.tensor(H_Rmax).cuda()
        if reference_FLAG:
            return torch.transpose(Wi, dim0=0, dim1=1), i0, H_Rmax
        else:
            return shape_tuple, xdim, ydim, Hiv, H_Rmax

    @staticmethod
    def background_filter_judegment(name, threshold=210, ratio=0.5):
        img = Image.open(name)
        img = img.convert('L')
        img = np.array(img)
        bkg_num = np.sum(img > threshold)
        bkg_cut = len(img.flatten()) * ratio
        if bkg_num > bkg_cut:
            return True
        else:
            return False

    def run_batch_colornorm(self, filenames, output_direc):
        print("To be normalized:", len(filenames[1:]), "using", filenames[0])

        tar_Wi, tar_i0, Hta_Rmax = self.get_params_on_images(filenames[0], reference_FLAG=True)

        for filename in filenames[1:]:
            if self.background_filter_judegment(filename):
                continue

            base_s = os.path.basename(filename).split('.')[0]  # source
            save_norm_name = os.path.join(output_direc, base_s + "_norm.png")
            #save_norm_name = filename
            print(save_norm_name)

            img_shape, xdim, ydim, Hiv, H_Rmax = self.get_params_on_images(filename, reference_FLAG=False)

            print("Color Normalization in progress...")
            norm_fac = torch.div(Hta_Rmax, H_Rmax)

            if (xdim * ydim) <= (self._max * self._max):
                Hsonorm = Hiv * norm_fac
                source_norm = tar_i0 * torch.exp(-torch.reshape(torch.matmul(Hsonorm, tar_Wi), img_shape))
                source_norm = source_norm.type(torch.uint8)
                cv2.imwrite(save_norm_name, cv2.cvtColor(source_norm.cpu().numpy(), cv2.COLOR_RGB2BGR))

            else:
                sourcenorm = np.memmap('wsi', dtype='uint8', mode='w+', shape=(ydim, xdim, 3))

                count = 0
                ind = 0
                np_max = 1000

                x_max = xdim
                y_max = min(max(int(np_max * np_max / x_max), 1), ydim)
                x_tl = range(0, xdim, x_max)
                y_tl = range(0, ydim, y_max)
                print("Patch-wise color normalization in progress...")
                total = len(x_tl) * len(y_tl)

                prev_progress = 0
                for x in x_tl:
                    for y in y_tl:
                        count += 1
                        xx = min(x_max, xdim - x)
                        yy = min(y_max, ydim - y)
                        pix = xx * yy
                        sh = np.array([yy, xx, 3])

                        # Back projection into spatial intensity space (Inverse Beer-Lambert space)
                        Hsonorm = torch.tensor(self.Hiv_temp[ind:ind + pix, :]).cuda() * norm_fac
                        source_norm = tar_i0 * torch.exp(-torch.reshape(torch.matmul(Hsonorm, tar_Wi), tuple(sh)))
                        sourcenorm[y:y + yy, x:x + xx, :3] = source_norm.type(torch.uint8).cpu().numpy()

                        ind += pix
                        percent = 5 * int(count * 20 / total)  # nearest 5 percent
                        if percent > prev_progress and percent < 100:
                            print(str(percent) + " percent complete..."),
                            prev_progress = percent
                print("Color Normalization complete!"),

                print("Saving normalized image...")
                pyimg = self.numpy2vips(sourcenorm)
                pyimg.tiffsave(save_norm_name.replace('png', 'svs'), tile=True, compression='lzw', xres=5000, yres=5000,
                               bigtiff=True, pyramid=True, Q=100)
                # xres and yres should be controlled to produce finer or coarser tif
                del sourcenorm
                del self.Hiv_temp

            if os.path.exists("H_target"):
                os.remove("H_target")
            if os.path.exists("H_source"):
                os.remove("H_source")
            if os.path.exists("wsi"):
                os.remove("wsi")
