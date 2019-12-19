import base64
import io
import json
import os.path as osp
import numpy as np
import PIL.Image

from labelme._version import __version__
from labelme.logger import logger
from labelme import PY2
from labelme import QT4
from labelme import utils


class LabelFileError(Exception):
    pass


class LabelFile(object):

    suffix = '.json'

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error('Failed opening image file: {}'.format(filename))
            return

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2 and QT4:
                format = 'PNG'
            elif ext in ['.jpg', '.jpeg']:
                format = 'JPEG'
            else:
                format = 'PNG'
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = [
            'imageData',
            'imagePath',
            'lineColor',
            'fillColor',
            'shapes',  # polygonal annotations
            'flags',   # image level flags
            'imageHeight',
            'imageWidth',
        ]
        try:
            with open(filename, 'rb' if PY2 else 'r') as f:
                data = json.load(f)
            if data['imageData'] is not None:
                imageData = base64.b64decode(data['imageData'])
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            else:
                # relative path from label file to relative path from cwd
                imagePath = osp.join(osp.dirname(filename), data['imagePath'])
                imageData = self.load_image_file(imagePath)
            flags = data.get('flags') or {}
            imagePath = data['imagePath']
            self._check_image_height_and_width(
                base64.b64encode(imageData).decode('utf-8'),
                data.get('imageHeight'),
                data.get('imageWidth'),
            )
            lineColor = data['lineColor']
            fillColor = data['fillColor']
            shapes = (
                (
                    s['label'],
                    s['points'],
                    s['line_color'],
                    s['fill_color'],
                    s.get('shape_type', 'polygon'),
                    s.get('flags', {}),
                )
                for s in data['shapes']
            )
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.lineColor = lineColor
        self.fillColor = fillColor
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                'imageHeight does not match with imageData or imagePath, '
                'so getting imageHeight from actual image.'
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                'imageWidth does not match with imageData or imagePath, '
                'so getting imageWidth from actual image.'
            )
            imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        lineColor=None,
        fillColor=None,
        otherData=None,
        flags=None,
    ):
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode('utf-8')
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            lineColor=lineColor,
            fillColor=fillColor,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            data[key] = value
        try:
            with open(filename, 'wb' if PY2 else 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)


        ## output cor file and json file  raocong
        if len(shapes) != 2:
            return
        l1 = shapes[0]['points']
        l2 = shapes[1]['points']
        if len(l1)!=len(l2):
            return
        if l1[0][1] < l2[0][1]:
            points = np.concatenate((l1,l2),axis = 1)
        else:
            points = np.concatenate((l2, l1), axis=1)
        cor_file = osp.splitext(filename)[0] + '_cor.txt'
        cor_points = np.round(points).reshape(-1,2)
        np.savetxt(cor_file, cor_points, fmt='%i %i')
        cor_points[:, 0] /= imageWidth
        cor_points[:, 1] /= imageHeight
        new_dict = {'z0': 50, 'z1': -90, 'uv': cor_points.tolist()}
        cor_json_file = osp.splitext(filename)[0] + '_cor.json'
        with open(cor_json_file, "w") as f:
            json.dump(new_dict, f)
        ## output cor file and json file  raocong


    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
