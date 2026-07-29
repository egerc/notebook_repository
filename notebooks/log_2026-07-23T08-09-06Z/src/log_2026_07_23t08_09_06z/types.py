from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import assert_never

from numpy import intp, number
from numpy.typing import NDArray

type NumericArray = NDArray[number]
type IndexArray = NDArray[intp]


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


def unwrap_result[A, E: Exception](
    result: Result[A, E], error_message: str | None = None
) -> A:
    match result:
        case Ok(value):
            return value
        case Err(reason):
            raise (reason if error_message is None else ValueError(error_message))
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
