from dataclasses import dataclass


@dataclass(frozen=True)
class Ok[A]:
    value: A


@dataclass(frozen=True)
class Err[E: Exception]:
    reason: E


class Just[A]:
    value: A


class Null:
    pass


type Maybe[A] = Just[A] | Null
type Result[A, E: Exception] = Ok[A] | Err[E]


def unwrap[A, E: Exception](result: Result[A, E]) -> A:
    match result:
        case Ok(value):
            return value
        case Err(reason):
            raise reason
