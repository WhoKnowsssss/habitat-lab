#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import inspect
import numbers
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray, numbers.Real]
DictTree = Dict[str, Union[TensorLike, "DictTree"]]
TensorIndexType = Union[int, slice, Tuple[Union[int, slice], ...]]

T = TypeVar("T")


_MapFuncType = Union[
    Callable[[T], T],
    Callable[[T, str], T],
]


class _TreeDict(Dict[str, Union["_TreeDict[T]", T]]):
    r"""A dictionary of tensors that can be indexed like a tensor or like a dictionary.

    .. code:: py
        t = TensorDict(a=torch.randn(2, 2), b=TensorDict(c=torch.randn(3, 3)))

        print(t)

        print(t[0, 0])

        print(t["a"])

    """

    @staticmethod
    def _is_instance(v: Any) -> bool:
        raise NotImplemented()

    @staticmethod
    def _to_instance(v: Any) -> T:
        raise NotImplemented()

    @classmethod
    def from_tree(cls, tree: DictTree) -> _TreeDict[T]:
        res = cls()
        for k, v in tree.items():
            if isinstance(v, dict):
                res[k] = cls.from_tree(v)
            elif cls._is_instance(v):
                res[k] = v
            else:
                res[k] = cls._to_instance(v)

        return res

    def to_tree(self) -> DictTree:
        res: DictTree = dict()
        for k, v in self.items():
            if isinstance(v, _TreeDict):
                res[k] = v.to_tree()
            else:
                res[k] = v

        return res

    @classmethod
    def from_flattened(
        cls, spec: List[Tuple[str, ...]], tensors: List[TensorLike]
    ) -> _TreeDict[T]:
        sort_ordering = sorted(range(len(spec)), key=lambda i: spec[i])
        spec = [spec[i] for i in sort_ordering]
        tensors = [tensors[i] for i in sort_ordering]

        res = cls()
        remaining = ([], [])
        for i, (k, *others), v in zip(range(len(spec)), spec, tensors):
            if len(others) == 0:
                if not cls._is_instance(v):
                    v = cls._to_instance(v)

                res[k] = v
            else:
                remaining[0].append(others)
                remaining[1].append(v)

            if ((i == len(spec) - 1) or k != spec[i + 1][0]) and len(
                remaining[0]
            ) > 0:
                res[k] = cls.from_flattened(*remaining)
                remaining = ([], [])

        return res

    def flatten(self) -> Tuple[List[Tuple[str, ...]], List[TensorLike]]:
        spec = []
        tensors = []
        for k, v in self.items():
            if isinstance(v, _TreeDict):
                for subk, subv in zip(*v.flatten()):
                    spec.append((k, *subk))
                    tensors.append(subv)
            else:
                spec.append((k,))
                tensors.append(v)

        return spec, tensors

    @overload
    def __getitem__(self, index: str) -> Union[_TreeDict[T], T]:
        ...

    @overload
    def __getitem__(self, index: TensorIndexType) -> _TreeDict[T]:
        ...

    def __getitem__(
        self, index: Union[str, TensorIndexType]
    ) -> Union[_TreeDict[T], T]:
        if isinstance(index, str):
            return super().__getitem__(index)
        else:
            return type(self)((k, v[index]) for k, v in self.items())

    @overload
    def set(
        self,
        index: str,
        value: Union[TensorLike, _TreeDict[T], DictTree],
        strict: bool = True,
    ) -> None:
        ...

    @overload
    def set(
        self,
        index: TensorIndexType,
        value: Union[_TreeDict[T], DictTree],
        strict: bool = True,
    ) -> None:
        ...

    def set(
        self,
        index: Union[str, TensorIndexType],
        value: Union[TensorLike, _TreeDict[T]],
        strict: bool = True,
    ) -> None:
        if isinstance(index, str):
            super().__setitem__(index, value)
        else:
            if strict and (self.keys() != value.keys()):
                raise KeyError(
                    "Keys don't match: Dest={} Source={}".format(
                        self.keys(), value.keys()
                    )
                )

            for k in self.keys():
                if k not in value:
                    if strict:
                        raise KeyError(f"Key {k} not in new value dictionary")
                    else:
                        continue

                v = value[k]
                dst = self[k]

                if isinstance(v, (_TreeDict, dict)):
                    dst.set(index, v, strict=strict)
                else:
                    if not self._is_instance(v):
                        v = self._to_instance(v)

                    dst[index] = v

    def __setitem__(
        self,
        index: Union[str, TensorIndexType],
        value: Union[torch.Tensor, _TreeDict[T]],
    ):
        self.set(index, value)

    @classmethod
    def map_func(
        cls,
        func: _MapFuncType,
        src: Union[_TreeDict[T], DictTree],
        dst: Optional[_TreeDict[T]] = None,
    ) -> _TreeDict[T]:
        if dst is None:
            dst = cls()

        sig = inspect.signature(func)
        sig_n_params = len(sig.parameters)
        for k, v in src.items():
            if isinstance(v, (cls, dict)):
                dst[k] = cls.map_func(func, v, dst.get(k, None))
            else:
                if isinstance(v, (tuple, list)):
                    args = v
                else:
                    args = (v,)

                if sig_n_params > len(args):
                    args = (*args, k)

                dst[k] = func(*args)

        return dst

    def map(self, func: _MapFuncType) -> _TreeDict[T]:
        return self.map_func(func, self)

    def map_in_place(self, func: _MapFuncType) -> _TreeDict[T]:
        return self.map_func(func, self, self)

    @classmethod
    def zip_func(
        cls,
        *trees: Iterable[Union[_TreeDict[T], dict]],
    ) -> _TreeDict[T]:
        keys = set(trees[0].keys())
        if not all(set(t.keys()).issubset(keys) for t in trees):
            raise RuntimeError(
                "All keys in dest(s) are {}. Frist tree has keys {}.".format(
                    set(k for t in trees[1:] for k in t.keys()),
                    keys,
                )
            )

        if not all(keys.issubset(t.keys()) for t in trees):
            raise RuntimeError(
                "Frist tree has keys {}.  All keys in dest(s) are {}.".format(
                    keys,
                    set(k for t in trees[1:] for k in t.keys()),
                )
            )

        res = cls()
        for k in keys:
            if isinstance(trees[0][k], (cls, dict)):
                res[k] = cls.zip_func(
                    *(t[k] for t in trees),
                )
            else:
                res[k] = tuple(t[k] for t in trees)

        return res

    def zip(
        self,
        *others: Iterable[Union[_TreeDict[T], dict]],
    ) -> _TreeDict[Tuple[T, ...]]:
        return self.zip_func(
            self,
            *others,
        )

    def slice_keys(
        self, *keys: Iterable[Union[str, Iterable[str]]]
    ) -> _TreeDict[T]:
        res = type(self)()
        for _k in keys:
            for k in (_k,) if isinstance(_k, str) else _k:
                assert k in self, f"Key {k} not in self"
                res[k] = self[k]

        return res

    def __deepcopy__(self, _memo=None) -> _TreeDict[T]:
        return self.from_tree(copy.deepcopy(self.to_tree(), memo=_memo))


class TensorDict(_TreeDict[torch.Tensor]):
    @staticmethod
    def _is_instance(v: Any) -> bool:
        return isinstance(v, torch.Tensor)

    @classmethod
    def _to_instance(cls, v: Any) -> torch.Tensor:
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v)
        else:
            return torch.as_tensor(v)

    def numpy(self) -> NDArrayDict:
        return NDArrayDict.from_tree(self)


class NDArrayDict(_TreeDict[np.ndarray]):
    @staticmethod
    def _is_instance(v: Any) -> bool:
        return isinstance(v, np.ndarray)

    @classmethod
    def _to_instance(cls, v: Any) -> np.ndarray:
        if isinstance(v, torch.Tensor):
            return v.numpy()
        else:
            return np.asarray(v)

    def torch(self) -> TensorDict:
        return TensorDict.from_tree(self)
