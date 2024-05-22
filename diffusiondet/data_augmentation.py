
import inspect
import pprint
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, List, Optional, TypeVar, Tuple, Union
import sys
from PIL import Image
import torch.nn.functional as F

import numpy as np
import torch


class Transform(metaclass=ABCMeta):

    def _set_attributes(self, params: Optional[List[Any]] = None) -> None:
        """
        Set attributes from the input list of parameters.

        Args:
            params (list): list of parameters.
        """

        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def apply_image(self, left_img: np.ndarray, right_image: np.ndarray):
        """
        Apply the transform on an image.

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: image after apply the transformation.
        """

    @abstractmethod
    def apply_coords(self, coords: np.ndarray):
        """
        Apply the transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: coordinates after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].
            This function should correctly transform coordinates outside the image as well.
        """

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply the transform on a full-image segmentation.
        By default will just perform "apply_image".

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
            or bool dtype.

        Returns:
            ndarray: segmentation after apply the transformation.
        """
        return self.apply_image(segmentation)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box. By default will transform
        the corner points and use their minimum/maximum to create a new
        axis-aligned box. Note that this default may change the size of your
        box, e.g. after rotations.

        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.
        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].

            This function does not clip boxes to force them inside the image.
            It is up to the application that uses the boxes to decide.
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()

        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords)
        coords = coords.reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)

        return trans_boxes

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply the transform on a list of polygons, each represented by a Nx2
        array. By default will just transform all the points.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            list[ndarray]: polygon after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        return [self.apply_coords(p) for p in polygons]

    @classmethod
    def register_type(cls, data_type: str, func: Optional[Callable] = None):
        """
        Register the given function as a handler that this transform will use
        for a specific data type.

        Args:
            data_type (str): the name of the data type (e.g., box)
            func (callable): takes a transform and a data, returns the
                transformed data.

        Examples:

        .. code-block:: python

            # call it directly
            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            HFlipTransform.register_type("voxel", func)

            # or, use it as a decorator
            @HFlipTransform.register_type("voxel")
            def func(flip_transform, voxel_data):
                return transformed_voxel_data

            # ...
            transform = HFlipTransform(...)
            transform.apply_voxel(voxel_data)  # func will be called
        """
        if func is None:  # the decorator style

            def wrapper(decorated_func):
                assert decorated_func is not None
                cls.register_type(data_type, decorated_func)
                return decorated_func

            return wrapper

        assert callable(
            func
        ), "You can only register a callable to a Transform. Got {} instead.".format(
            func
        )
        argspec = inspect.getfullargspec(func)
        assert len(argspec.args) == 2, (
            "You can only register a function that takes two positional "
            "arguments to a Transform! Got a function with spec {}".format(str(argspec))
        )
        setattr(cls, "apply_" + data_type, func)

    def inverse(self) -> "Transform":
        """
        Create a transform that inverts the geometric changes (i.e. change of
        coordinates) of this transform.

        Note that the inverse is meant for geometric changes only.
        The inverse of photometric transforms that do not change coordinates
        is defined to be a no-op, even if they may be invertible.

        Returns:
            Transform:
        """
        raise NotImplementedError

    def __repr__(self):
        """
        Produce something like:
        "MyTransform(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL
                    and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(
                        name
                    )
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                attr_str = pprint.pformat(attr)
                if "\n" in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    attr_str = "..."
                argstr.append("{}={}".format(name, attr_str))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()


def _get_aug_input_args(aug, aug_input) -> List[Any]:
    args = []
    args.append(getattr(aug_input, "left_image"))
    return args


class Augmentation:

    input_args: Optional[Tuple[str]] = None

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def get_transform(self, *args) -> Transform:
        raise NotImplementedError

    def __call__(self, aug_input) -> Transform:
        args = _get_aug_input_args(self, aug_input)
        tfm = self.get_transform(*args)
        assert isinstance(tfm, (Transform, TransformList)), (
            f"{type(self)}.get_transform must return an instance of Transform! "
            f"Got {type(tfm)} instead."
        )
        aug_input.transform(tfm)
        return tfm

    def _rand_range(self, low=1.0, high=None, size=None):
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                attr_str = pprint.pformat(attr)
                if "\n" in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    attr_str = "..."
                argstr.append("{}={}".format(name, attr_str))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__

_T = TypeVar("_T")

class TransformList(Transform):
    """
    Maintain a list of transform operations which will be applied in sequence.
    Attributes:
        transforms (list[Transform])
    """

    def __init__(self, transforms: List[Transform]):
        """
        Args:
            transforms (list[Transform]): list of transforms to perform.
        """
        super().__init__()
        # "Flatten" the list so that TransformList do not recursively contain TransfomList.
        # The additional hierarchy does not change semantic of the class, but cause extra
        # complexities in e.g, telling whether a TransformList contains certain Transform
        tfms_flatten = []
        for t in transforms:
            assert isinstance(
                t, Transform
            ), f"TransformList requires a list of Transform. Got type {type(t)}!"
            if isinstance(t, TransformList):
                tfms_flatten.extend(t.transforms)
            else:
                tfms_flatten.append(t)
        self.transforms = tfms_flatten

    def _apply(self, x: _T, meth: str) -> _T:
        """
        Apply the transforms on the input.
        Args:
            x: input to apply the transform operations.
            meth (str): meth.
        Returns:
            x: after apply the transformation.
        """
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def __add__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(self.transforms + others)

    def __iadd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        self.transforms.extend(others)
        return self

    def __radd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(others + self.transforms)

    def __len__(self) -> int:
        """
        Returns:
            Number of transforms contained in the TransformList.
        """
        return len(self.transforms)

    def __getitem__(self, idx) -> Transform:
        return self.transforms[idx]

    def inverse(self) -> "TransformList":
        """
        Invert each transform in reversed order.
        """
        return TransformList([x.inverse() for x in self.transforms[::-1]])

    def __repr__(self) -> str:
        msgs = [str(t) for t in self.transforms]
        return "TransformList[{}]".format(", ".join(msgs))

    __str__ = __repr__

    # The actual implementations are provided in __getattribute__.
    # But abstract methods need to be declared here.
    def apply_coords(self, coords):
        for t in self.transforms:
            coords = t.apply_coords(coords)
        return coords

    def apply_box_(self, bbox):
        for t in self.transforms:
            bbox = t.apply_box(bbox)
        return bbox

    def apply_image(self, x):
        raise NotImplementedError


class NoOpTransform(Transform):
    """
    A transform that does nothing.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, left_img: np.ndarray) -> [np.ndarray, np.ndarray]:
        return left_img

    def apply_coords(self, coords: np.ndarray) -> [np.ndarray, np.ndarray]:
        return coords
    
    def apply_scale(self, scale):
        return scale

    def check_flip(self, ind):
        if ind:
            return True
        return False


class HFlipTransform(Transform):
    """
    Perform horizontal flip.
    """

    def __init__(self, width: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, left_img: np.ndarray) -> [np.ndarray, np.ndarray]:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        if left_img.ndim <= 3:  # HxW, HxWxC
            return np.flip(left_img, axis=1)
        else:
            return np.flip(left_img, axis=-2)

    def apply_coords(self, coords: np.ndarray) -> [np.ndarray, np.ndarray]:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 0] = self.width - coords[:, 0]
        return coords
    
    def apply_scale(self, scale):
        return scale

    def check_flip(self, ind):
        return True


class CropTransform(Transform):
    def __init__(
        self,
        x0: int,
        y0: int,
        w: int,
        h: int,
        orig_w: Optional[int] = None,
        orig_h: Optional[int] = None,
    ):
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
            orig_w, orig_h (int): optional, the original width and height
                before cropping. Needed to make this transform invertible.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, left_img: np.ndarray) -> [np.ndarray, np.ndarray]:
        """
        Crop the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: cropped image(s).
        """
        if len(left_img.shape) <= 3:
            return left_img[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]
        else:
            return left_img[..., self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w, :]

    def apply_coords(self, coords: np.ndarray) -> [np.ndarray, np.ndarray]:
        """
        Apply crop transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_scale(self, scale):
        return scale
    
    def check_flip(self, ind):
        if ind:
            return True
        return False


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, left_image, interp=None):
        assert left_image.shape[:2] == (self.h, self.w)
        assert len(left_image.shape) <= 4
        interp_method = interp if interp is not None else self.interp

        if left_image.dtype == np.uint8:
            if len(left_image.shape) > 2 and left_image.shape[2] == 1:
                left_pil_image = Image.fromarray(left_image[:, :, 0], mode="L")
            else:
                left_pil_image = Image.fromarray(left_image)

            left_pil_image = left_pil_image.resize((self.new_w, self.new_h), interp_method)
            left_ret = np.asarray(left_pil_image)

            if len(left_image.shape) > 2 and left_image.shape[2] == 1:
                left_ret = np.expand_dims(left_ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in left_image.strides):
                left_image = np.ascontiguousarray(left_image)
            left_image = torch.from_numpy(left_image)
            shape = list(left_image.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            left_image = left_image.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            left_image = F.interpolate(
                left_image, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            left_ret = left_image.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return left_ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords
    
    def apply_scale(self, scale):
        scale = scale * (self.new_w * 1.0 / self.w)
        return scale
    
    def check_flip(self, ind):
        if ind:
            return True
        return False


class RandomFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, left_image):
        h, w = left_image.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class RandomCrop(Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.

        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        # TODO style of relative_range and absolute_range are not consistent:
        # one takes (h, w) but another takes (min, max)
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())

    def get_transform(self, left_image):
        h, w = left_image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError("Unknown crop type {}".format(self.crop_type))


class _TransformToAug(Augmentation):
    def __init__(self, tfm: Transform):
        self.tfm = tfm

    def get_transform(self, *args):
        return self.tfm

    def __repr__(self):
        return repr(self.tfm)

    __str__ = __repr__


def _transform_to_aug(tfm_or_aug):
    """
    Wrap Transform into Augmentation.
    Private, used internally to implement augmentations.
    """
    assert isinstance(tfm_or_aug, (Transform, Augmentation)), tfm_or_aug
    if isinstance(tfm_or_aug, Augmentation):
        return tfm_or_aug
    else:
        return _TransformToAug(tfm_or_aug)


class ResizeShortestEdge(Augmentation):
    """
    Resize the image while keeping the aspect ratio unchanged.
    It attempts to scale the shorter edge to the given `short_edge_length`,
    as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    @torch.jit.unused
    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._init(locals())

    @torch.jit.unused
    def get_transform(self, left_image):
        h, w = left_image.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        newh, neww = ResizeShortestEdge.get_output_shape(h, w, size, self.max_size)
        return ResizeTransform(h, w, newh, neww, self.interp)

    @staticmethod
    def get_output_shape(
        oldh: int, oldw: int, short_edge_length: int, max_size: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class AugmentationList(Augmentation):
    """
    Apply a sequence of augmentations.

    It has ``__call__`` method to apply the augmentations.

    Note that :meth:`get_transform` method is impossible (will throw error if called)
    for :class:`AugmentationList`, because in order to apply a sequence of augmentations,
    the kth augmentation must be applied first, to provide inputs needed by the (k+1)th
    augmentation.
    """

    def __init__(self, augs):
        """
        Args:
            augs (list[Augmentation or Transform]):
        """
        super().__init__()
        self.augs = [_transform_to_aug(x) for x in augs]

    def __call__(self, input) -> TransformList:
        tfms = []
        for x in self.augs:
            tfm = x(input)
            tfms.append(tfm)
        return TransformList(tfms)

    def __repr__(self):
        msgs = [str(x) for x in self.augs]
        return "AugmentationList[{}]".format(", ".join(msgs))

    __str__ = __repr__


def _check_img_dtype(img):
    assert isinstance(img, np.ndarray), "[Augmentation] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[Augmentation] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [2, 3], img.ndim


class AugInput:

    # TODO maybe should support more builtin data types here
    def __init__(
        self,
        left_image: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
    ):

        _check_img_dtype(left_image)
        self.left_image = left_image
        self.boxes = boxes
        self.sem_seg = sem_seg

    def transform(self, tfm: Transform) -> None:

        self.left_image = tfm.apply_image(self.left_image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)

    def apply_augmentations(
        self, augmentations: List[Union[Augmentation, Transform]]
    ) -> TransformList:
        """
        Equivalent of ``AugmentationList(augmentations)(self)``
        """
        return AugmentationList(augmentations)(self)


def apply_augmentations(augmentations: List[Union[Transform, Augmentation]], left_image):
    """
    Use ``T.AugmentationList(augmentations)(inputs)`` instead.
    """
    if isinstance(left_image, np.ndarray):
        # handle the common case of image-only Augmentation, also for backward compatibility
        image_only = True
        inputs = AugInput(left_image)
    else:
        image_only = False
    tfms = inputs.apply_augmentations(augmentations)

    return inputs.left_image, tfms
