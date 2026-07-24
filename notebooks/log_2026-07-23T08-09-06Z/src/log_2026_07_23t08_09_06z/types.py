from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


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
class Null:
    pass


type Maybe[A] = Just[A] | Null
type Result[A, E: Exception] = Ok[A] | Err[E]


def optional_to_maybe[A](value: A | None) -> Maybe[A]:
    return Just(value) if value is not None else Null()


def bind_maybe[A, B](maybe_a: Maybe[A], func: Callable[[A], Maybe[B]]) -> Maybe[B]:
    match maybe_a:
        case Just(value):
            return func(value)
        case Null():
            return Null()


def bind_result[A, B, E: Exception](
    result: Result[A, E], func: Callable[[A], Result[B, E]]
) -> Result[B, E]:
    match result:
        case Ok(value):
            return func(value)
        case Err(reason):
            return Err(reason)


def unwrap_result[A, E: Exception](result: Result[A, E]) -> A:
    match result:
        case Ok(value):
            return value
        case Err(reason):
            raise reason


def unwrap_maybe[A](maybe_a: Maybe[A]) -> A:
    match maybe_a:
        case Just(value):
            return value
        case Null():
            raise ValueError("unwrap_maybe: Null")
