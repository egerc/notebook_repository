from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from logging import Logger
from typing import assert_never

from anndata.typing import AnnData  # type: ignore
from numpy import intp, number
from numpy.typing import NDArray
from pydantic.types import NonNegativeInt, PositiveInt

type NumericArray = NDArray[number]
type IndexArray = NDArray[intp]


class SamplingSplit(StrEnum):
    TRAIN = auto()
    TEST = auto()


@dataclass(frozen=True)
class DatasetSplit:
    cell_split: SamplingSplit
    gene_split: SamplingSplit


type EitherOrBoth[A, B] = A | B | tuple[A, B]


type EvaluationResult = dict[DatasetSplit, Result[tuple[AnnData, AnnData], Exception]]


@dataclass(frozen=True, slots=True)
class DownsamplingConfig:
    value: PositiveInt
    seed: NonNegativeInt


@dataclass(frozen=True, slots=True)
class Ok[A]:
    value: A


@dataclass(frozen=True, slots=True)
class Err[E: Exception]:
    reason: E


@dataclass(frozen=True, slots=True)
class Just[A]:
    value: A


@dataclass(frozen=True, slots=True)
class Nothing:
    pass


type Maybe[A] = Just[A] | Nothing
type Result[A, E: Exception] = Ok[A] | Err[E]


def optional_to_maybe[A](value: A | None) -> Maybe[A]:
    return Just(value) if value is not None else Nothing()


def bind_maybe[A, B](maybe_a: Maybe[A], func: Callable[[A], Maybe[B]]) -> Maybe[B]:
    match maybe_a:
        case Just(value):
            return func(value)
        case Nothing():
            return Nothing()
        case _:
            assert_never(maybe_a)


def bind_result[A, B, E_1: Exception, E_2: Exception](
    result: Result[A, E_1], func: Callable[[A], Result[B, E_2]]
) -> Result[B, E_1 | E_2]:
    match result:
        case Ok(value):
            return func(value)
        case Err(reason):
            return Err(reason)
        case _:
            assert_never(result)


def map_result[A, B, E: Exception](
    result: Result[A, E], func: Callable[[A], B]
) -> Result[B, E]:
    match result:
        case Ok(value):
            return Ok(func(value))
        case Err(reason):
            return Err(reason)
        case _:
            assert_never(result)


def unwrap_result[A, E: Exception](
    result: Result[A, E], exception: Exception | None = None
) -> A:
    match result:
        case Ok(value):
            return value
        case Err(reason):
            if exception is not None:
                raise exception from reason
            raise reason
        case _:
            assert_never(result)


def unwrap_result_or_default[A, E: Exception](result: Result[A, E], default: A) -> A:
    match result:
        case Ok(value):
            return value
        case Err(_):
            return default
        case _:
            assert_never(result)


def zip_result[A, B, E1: Exception, E2: Exception](
    res_a: Result[A, E1],
    res_b: Result[B, E2],
) -> Result[tuple[A, B], E1 | E2]:
    """Combines two Results into a tuple Result. Short-circuits on the first Err (left-to-right)."""
    match res_a:
        case Ok(a):
            match res_b:
                case Ok(b):
                    return Ok((a, b))
                case Err(err_b):
                    return Err(err_b)
                case _:
                    assert_never(res_b)
        case Err(err_a):
            return Err(err_a)
        case _:
            assert_never(res_a)


def starmap_result[A, B, C, E: Exception](
    result: Result[tuple[A, B], E], func: Callable[[A, B], C]
) -> Result[C, E]:
    """Maps a two-argument function over a Result containing a 2-tuple by unpacking it."""
    match result:
        case Ok((a, b)):
            return Ok(func(a, b))
        case Err(reason):
            return Err(reason)
        case _:
            assert_never(result)


def starbind_result[A, B, C, E1: Exception, E2: Exception](
    result: Result[tuple[A, B], E1], func: Callable[[A, B], Result[C, E2]]
) -> Result[C, E1 | E2]:
    """Binds a function consuming unpacked tuple elements and returning a new Result."""
    match result:
        case Ok((a, b)):
            return func(a, b)
        case Err(reason):
            return Err(reason)
        case _:
            assert_never(result)


def safe_apply_single[A, B](value: A, func: Callable[[A], B]) -> Result[B, Exception]:
    try:
        return Ok(func(value))
    except Exception as e:  # noqa
        return Err(e)


def log_result[A, E: Exception](
    result: Result[A, E], logger: Logger, func: Callable[[A], str]
) -> None:
    match result:
        case Ok(value):
            logger.info(func(value))
        case Err(reason):
            logger.error(reason)


def result_is_ok[A, E: Exception](result: Result[A, E]) -> bool:
    match result:
        case Ok(_):
            return True
        case Err(_):
            return False
        case _:
            assert_never(result)


def ok_if[A, E: Exception](
    value: A, predicate: Callable[[A], bool], exception: E
) -> Result[A, E]:
    if predicate(value):
        return Ok(value)
    return Err(exception)


def map_err[T, E: Exception, E2: Exception](
    result: Result[T, E], fn: Callable[[E], E2]
) -> Result[T, E2]:
    match result:
        case Ok(val):
            return Ok(val)
        case Err(err):
            return Err(fn(err))
        case _:
            assert_never(result)


def ok_or[A, E: Exception](maybe_a: Maybe[A], exception: E) -> Result[A, E]:
    match maybe_a:
        case Just(value):
            return Ok(value)
        case Nothing():
            return Err(exception)
        case _:
            assert_never(maybe_a)


def maybe_from_optional[A](value: A | None) -> Maybe[A]:
    return Just(value) if value is not None else Nothing()


def unwrap_maybe[A](maybe_a: Maybe[A], exception: Exception | None = None) -> A:
    match maybe_a:
        case Just(value):
            return value
        case Nothing():
            raise (
                exception if exception is not None else ValueError("unwrap_maybe: Null")
            )
        case _:
            assert_never(maybe_a)


def rights[A, E: Exception](
    results: Sequence[Result[A, E] | Maybe[A]],
) -> Generator[A, None, None]:
    for result in results:
        match result:
            case Ok(value) | Just(value):
                yield value
            case Err(_) | Nothing():
                continue
            case _:
                assert_never(result)


def rights_default[A, E: Exception](
    results: Sequence[Result[A, E] | Maybe[A]],
    default: A,
) -> Generator[A, None, None]:
    for result in results:
        match result:
            case Ok(value) | Just(value):
                yield value
            case Err(_) | Nothing():
                yield default
            case _:
                assert_never(result)


def lefts[A, E: Exception](
    results: Sequence[Result[A, E] | Maybe[A]],
) -> Generator[Err[E] | Nothing, None, None]:
    for result in results:
        match result:
            case Ok(_) | Just(_):
                continue
            case Err() | Nothing():
                yield result
            case _:
                assert_never(result)
