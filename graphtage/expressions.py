from collections import deque
from enum import Enum
from io import StringIO
from typing import Any, Callable, Collection, Dict, Generic, IO, Iterable, Iterator, List, Optional, Set, \
    SupportsFloat, SupportsInt, Tuple, Type, TypeVar, Union
import itertools


DEFAULT_GLOBALS: Dict[str, Any] = {
    obj.__name__: obj for obj in (
        str, bool, int, bytes, float, bytearray, dict, set, frozenset,
        enumerate, zip, map, filter, any, all, chr, ord, abs, ascii, bin, bool, complex, hash, hex, oct, min, max, id,
        iter, len, list, slice, sorted, sum, tuple, round
    )
}


OPERATORS_BY_NAME: Dict[str, 'Operator'] = {}


class ParseError(RuntimeError):
    def __init__(self, message, offset):
        super().__init__(message)
        self.offset = offset

    def __str__(self):
        return f"{super().__str__()} at offset {self.offset}"


def function_call(obj, function_name):
    raise NotImplementedError("TODO: Implement")


def get_member(obj, member):
    if not isinstance(member, IdentifierToken):
        raise ParseError(f"member name expected, instead found {member}", member.offset)
    if member.name.startswith('_'):
        raise ParseError(f"Cannot read protected and private member variables: {obj}.{member.name}", member.offset)
    return getattr(obj, member.name)


class Operator(Enum):
    MEMBER_ACCESS = ('.', 1, lambda a, b: get_member(a, b), True, 2, False, (True, False))
    GETITEM = ('[', 1, lambda a, b: a[b])
    FUNCTION_CALL = ('→', 2, lambda a, b: a(*b), True, 2, True)
    UNARY_PLUS = ('+', 3, lambda a: a, False, 1, True)
    UNARY_MINUS = ('-', 3, lambda a: -a, False, 1, True)
    LOGICAL_NOT = ('not', 3, lambda a: not a, False, 1)
    BITWISE_NOT = ('~', 3, lambda a: ~a, False, 1)
    MULTIPLICATION = ('*', 4, lambda a, b: a * b)
    DIVISION = ('/', 4, lambda a, b: a / b)
    INT_DIVISION = ('//', 4, lambda a, b: a // b)
    REMAINDER = ('%', 4, lambda a, b: a % b)
    ADDITION = ('+', 5, lambda a, b: a + b)
    SUBTRACTION = ('-', 5, lambda a, b: a - b)
    BITWISE_LEFT_SHIFT = ('<<', 6, lambda a, b: a << b)
    BITWISE_RIGHT_SHIFT = ('>>', 6, lambda a, b: a >> b)
    LESS_THAN = ('<', 7, lambda a, b: a < b)
    GREATER_THAN = ('>', 7, lambda a, b: a > b)
    LESS_THAN_EQUAL = ('<=', 7, lambda a, b: a <= b)
    GREATER_THAN_EQUAL = ('>=', 7, lambda a, b: a >= b)
    EQUALS = ('==', 8, lambda a, b: a == b)
    NOT_EQUAL = ('!=', 8, lambda a, b: a != b)
    BITWISE_AND = ('&', 9, lambda a, b: a & b)
    BITWISE_XOR = ('^', 10, lambda a, b: a ^ b)
    BITWISE_OR = ('|', 11, lambda a, b: a | b)
    LOGICAL_AND = ('and', 12, lambda a, b: a and b)
    LOGICAL_OR = ('or', 13, lambda a, b: a or b)
    TERNARY_ELSE = (':', 14, lambda a, b: (a, b), False)
    TERNARY_CONDITIONAL = ('?', 15, lambda a, b: b[bool(a)], False)

    def __init__(self,
                 token: str,
                 priority: int,
                 execute: Callable[[Any, Any], Any],
                 is_left_associative: bool = True,
                 arity: int = 2,
                 multiple_arity: bool = False,
                 expand: Optional[Tuple[bool, ...]] = None):
        self.token: str = token
        self.priority: int = priority
        self.execute: Callable[[Any, Any], Any] = execute
        self.left_associative: bool = is_left_associative
        self.arity: int = arity
        self.multiple_arity: bool = multiple_arity
        if expand is None:
            self.expand: Tuple[bool, ...] = (True,) * self.arity
        else:
            self.expand: Tuple[bool, ...] = expand
        if not multiple_arity:
            OPERATORS_BY_NAME[self.token] = self


IDENTIFIER_BYTES: Set[str] = {
    chr(i) for i in range(ord('A'), ord('Z') + 1)
} | {
    chr(i) for i in range(ord('a'), ord('z') + 1)
} | {
    chr(i) for i in range(ord('0'), ord('9') + 1)
} | {
    '-', '_'
}


class Token:
    def __init__(self, raw_text: str, offset: int):
        self._raw: str = raw_text
        self.offset = offset

    @property
    def raw_token(self) -> str:
        return self._raw

    def __len__(self):
        return len(self._raw)

    def __str__(self):
        return self._raw

    def __repr__(self):
        return f"{self.__class__.__name__}({self._raw!r})"


class PairedToken:
    name: str = None


class PairedStartToken(PairedToken):
    discard: bool = True


class PairedEndToken(PairedToken):
    start_token_type: Type[PairedStartToken] = None


class Parenthesis(Token):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class OpenParen(Parenthesis, PairedStartToken):
    name = 'parenthesis'

    def __init__(self, offset: int, is_function_call: bool):
        super().__init__('(', offset)
        self.is_function_call = is_function_call


class CloseParen(Parenthesis, PairedEndToken):
    name = 'parenthesis'
    start_token_type = OpenParen

    def __init__(self, offset: int):
        super().__init__(')', offset)


class OperatorToken(Token):
    def __init__(self, op: Union[str, Operator], offset: int):
        if isinstance(op, str):
            op = OPERATORS_BY_NAME[op]
        super().__init__(op.token, offset)
        self.op: Operator = op

    def __repr__(self):
        return f"{self.__class__.__name__}(op={self.op!r})"


class OpenBracket(OperatorToken, PairedStartToken):
    name = 'brackets'
    discard = False

    def __init__(self, offset: int, is_list: bool):
        super().__init__(Operator.GETITEM, offset)
        self.is_list = is_list

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class CloseBracket(Token, PairedEndToken):
    name = 'brackets'
    start_token_type = OpenBracket

    def __init__(self, offset: int):
        super().__init__(']', offset)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Comma(Token):
    def __init__(self, offset: int):
        super().__init__(',', offset)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class FixedSizeCollection(Token):
    def __init__(self, size: int, container_type: Type[Collection], offset: int):
        super().__init__(container_type.__name__, offset)
        self.size: int = size
        self.container_type: Type[Collection] = container_type

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, container_type={self.container_type!r})"


class IdentifierToken(Token):
    def __init__(self, name: str, offset: int):
        super().__init__(name, offset)
        self.name: str = name


N = TypeVar('N', bound=Union[SupportsInt, SupportsFloat])


class NumericToken(Token, Generic[N]):
    def __init__(self, raw_str: str, value: N, offset: int):
        super().__init__(raw_str, offset)
        self.value: N = value

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return f"{self.__class__.__name__}(raw_str={self.raw_token!r}, value={self.value!r})"


class IntegerToken(NumericToken[int]):
    pass


class FloatToken(NumericToken[float]):
    pass


class StringToken(Token):
    pass


class FunctionCall(OperatorToken):
    def __init__(self, offset: int):
        super().__init__(Operator.FUNCTION_CALL, offset)


class Tokenizer:
    def __init__(self, stream: Union[str, IO]):
        if isinstance(stream, str):
            stream = StringIO(stream)
        self._stream: IO = stream
        self._buffer: deque = deque()
        self._next_token: Optional[Token] = None
        self.prev_token: Optional[Token] = None
        self._offset: int = 0
        self._function_parens: List[bool] = []

    def _peek_byte(self, n=1) -> str:
        bytes_needed = n - len(self._buffer)
        if bytes_needed > 0:
            b = self._stream.read(bytes_needed)
            self._buffer.extend(b)
        return ''.join(itertools.islice(self._buffer, n))

    def _pop_byte(self, n=1) -> str:
        if len(self._buffer) < n:
            ret = ''.join(self._buffer) + self._stream.read(1)
        else:
            ret = ''.join(self._buffer.popleft() for _ in range(n))
        assert len(ret) <= n
        self._offset += len(ret)
        return ret

    def peek(self) -> Optional[Token]:
        if self._next_token is not None:
            return self._next_token
        elif isinstance(self.prev_token, FunctionCall):
            return OpenParen(self.prev_token.offset, is_function_call=True)
        ret: Optional[Token] = None
        operand: Optional[str] = None
        string_start: Optional[str] = None
        string_start_pos: int = 0
        # ignore leading whitespace
        while self._peek_byte() == ' ' or self._peek_byte() == '\t':
            self._pop_byte()
        while ret is None:
            c: str = self._peek_byte(3)
            if len(c) == 0:
                break
            elif operand is not None or string_start is not None:
                if string_start:
                    if c[0] == '\\':
                        operand += self._pop_byte(2)[1]
                    elif c[0] == string_start:
                        self._pop_byte()
                        return StringToken(operand, self._offset - 1)
                    elif operand is None:
                        operand = self._pop_byte()
                    else:
                        operand += self._pop_byte()
                elif c[0] in IDENTIFIER_BYTES:
                    operand += self._pop_byte()
                else:
                    break
            elif operand is None and (c[0] == '"' or c[0] == "'"):
                string_start_pos = self._offset
                string_start = self._pop_byte()
                operand = ''
            elif c[0] == '[':
                # We have to check for this before parsing OperatorTokens otherwise we will yield an
                # OperatorToken('[') instead of an OpenBracket()

                # An opening bracket indicates creating a list (rather than a getitem) if it follows a comma
                # or '[' or '(' or an operator
                is_list = self.prev_token is None or isinstance(self.prev_token, Comma) \
                          or isinstance(self.prev_token, PairedStartToken) or isinstance(self.prev_token, OperatorToken)
                ret = OpenBracket(self._offset, is_list=is_list)
            elif c[0] == ',':
                # we also have to check this before the operators for the same reason as '[' above:
                ret = Comma(self._offset)
            elif c in OPERATORS_BY_NAME:
                ret = OperatorToken(c, offset=self._offset)
            elif c[:2] in OPERATORS_BY_NAME:
                ret = OperatorToken(c[:2], offset=self._offset)
            elif c[0] in OPERATORS_BY_NAME:
                if c[0] == '+':
                    if self.prev_token is None or isinstance(self.prev_token, OperatorToken):
                        ret = OperatorToken(Operator.UNARY_PLUS, offset=self._offset)
                    else:
                        ret = OperatorToken(Operator.ADDITION, offset=self._offset)
                elif c[0] == '-':
                    if self.prev_token is None or isinstance(self.prev_token, OperatorToken):
                        ret = OperatorToken(Operator.UNARY_MINUS, offset=self._offset)
                    else:
                        ret = OperatorToken(Operator.SUBTRACTION, offset=self._offset)
                else:
                    ret = OperatorToken(c[0], self._offset)
            elif c[0] == '(':
                if isinstance(self.prev_token, IdentifierToken) \
                        or isinstance(self.prev_token, PairedEndToken):
                    # Inject a function call before the parenthesis
                    # We will emit the OpenParen token on the next call to peek()
                    ret = FunctionCall(self._offset)
                else:
                    ret = OpenParen(self._offset, is_function_call=False)
            elif c[0] == ')':
                ret = CloseParen(self._offset)
            elif c[0] == ']':
                ret = CloseBracket(self._offset)
            elif c[0] == ' ' or c[0] == '\t':
                break
            else:
                operand = self._pop_byte()
        if string_start is not None:
            escaped_str = operand.replace(string_start, "\\" + string_start)
            raise ParseError(f'Unterminated string: {string_start}{escaped_str}...', string_start_pos)
        elif operand is not None:
            if operand.startswith('0x'):
                errored = False
                try:
                    ret = IntegerToken(operand, int(operand, 16), self._offset)
                except ValueError:
                    errored = True
                if errored:
                    raise ParseError("Malformed hex string", self._offset - len(operand))
            elif operand.startswith('0o'):
                errored = False
                try:
                    ret = IntegerToken(operand, int(operand, 8), self._offset)
                except ValueError:
                    errored = True
                if errored:
                    raise ParseError("Malformed octal string", self._offset - len(operand))
            elif operand.startswith('0b'):
                errored = False
                try:
                    ret = IntegerToken(operand, int(operand, 2), self._offset)
                except ValueError:
                    errored = True
                if errored:
                    raise ParseError("Malformed binary string", self._offset - len(operand))
            else:
                # Is this an integer?
                try:
                    ret = IntegerToken(operand, int(operand), self._offset)
                except ValueError:
                    # Maybe it is a float?
                    try:
                        ret = FloatToken(operand, float(operand), self._offset)
                    except ValueError:
                        ret = IdentifierToken(operand, self._offset)
        elif ret is not None:
            self._pop_byte(len(ret))
        return ret

    def has_next(self) -> bool:
        return self.peek() is not None

    def next(self) -> Optional[Token]:
        ret = self.peek()
        self.prev_token = ret
        self._next_token = None
        return ret

    def __iter__(self) -> Iterator[Token]:
        while True:
            ret = self.next()
            if ret is None:
                break
            yield ret


def tokenize(stream_or_str) -> Iterator[Token]:
    yield from Tokenizer(stream_or_str)


class CollectionInfo:
    def __init__(self, collection_type: Union[Type[tuple], Type[list]]):
        self.num_commas: int = 0
        self.collection_type: Union[Type[tuple], Type[list]] = collection_type
        self.last_was_comma: bool = False


def infix_to_rpn(tokens: Iterable[Token]) -> Iterator[Token]:
    """Converts an infix expression to reverse Polish notation using the Shunting Yard algorithm"""
    operators: List[OperatorToken] = []
    collection_stack: List[CollectionInfo] = []

    for token in tokens:
        if isinstance(token, Comma):
            if not collection_stack:
                raise ParseError("Unexpected comma outside of parenthesis or brackets", token.offset)
            collection_stack[-1].num_commas += 1
            collection_stack[-1].last_was_comma = True
            continue
        elif isinstance(token, PairedStartToken):
            operators.append(token)
            if isinstance(token, OpenParen):
                collection_stack.append(CollectionInfo(tuple))
            elif isinstance(token, OpenBracket):
                collection_stack.append(CollectionInfo(list))
            else:
                raise NotImplementedError(f"Add support for tokens of type {type(token)}")
        elif isinstance(token, PairedEndToken):
            looking_for: Type[PairedStartToken] = token.start_token_type
            while operators and not isinstance(operators[-1], looking_for):
                if isinstance(operators[-1], PairedStartToken):
                    raise ParseError(f"{token.name} mismatched with a {operators[-1].name}", token.offset)
                yield operators.pop()
            if not operators:
                raise ParseError(f"Mismatched {token.name}", token.offset)
            elif isinstance(operators[-1], looking_for):
                assert len(collection_stack) > 0
                if isinstance(operators[-1], OpenBracket):
                    if collection_stack[-1].last_was_comma:
                        raise ParseError("Unexpected comma at end of list", token.offset)
                    if operators[-1].is_list and collection_stack[-1].num_commas == 0:
                        # This opening bracket was used in a context that implies list creation rather than getitem
                        collection_stack[-1].num_commas = 1
                        collection_stack[-1].last_was_comma = True
                elif isinstance(operators[-1], OpenParen):
                    if operators[-1].is_function_call and collection_stack[-1].num_commas == 0:
                        # Force function argument lists to be treated as tuples
                        collection_stack[-1].last_was_comma = True
                        collection_stack[-1].num_commas = 1
                if collection_stack[-1].num_commas > 0:
                    size = collection_stack[-1].num_commas
                    if not collection_stack[-1].last_was_comma:
                        size += 1
                    yield FixedSizeCollection(size, collection_stack[-1].collection_type, operators[-1].offset)
                    collection_stack.pop()
                    operators.pop()
                elif looking_for.discard:
                    operators.pop()
                else:
                    yield operators.pop()
        elif isinstance(token, OperatorToken):
            while operators and not isinstance(operators[-1], PairedStartToken) and \
                    (operators[-1].op.priority < token.op.priority
                        or
                        (operators[-1].op.priority == token.op.priority and operators[-1].op.left_associative)):
                yield operators.pop()
            operators.append(token)
        else:
            yield token
        if collection_stack:
            collection_stack[-1].last_was_comma = False

    while operators:
        top = operators.pop()
        if isinstance(top, PairedToken):
            raise RuntimeError(f"Mismatched {top.name}")
        yield top


class Expression:
    def __init__(self, rpn: Iterable[Token]):
        self.tokens: Tuple[Token, ...] = tuple(rpn)

    @staticmethod
    def get_value(token: Token, locals: Dict[str, Any], globals: Dict[str, Any]):
        if isinstance(token, NumericToken):
            return token.value
        elif isinstance(token, StringToken):
            return token.raw_token
        elif isinstance(token, IdentifierToken):
            if token.name in locals:
                return locals[token.name]
            elif token.name in globals:
                return globals[token.name]
            else:
                raise KeyError(f'Unknown identifier {token.name}')
        elif isinstance(token, Token):
            raise ValueError(f"Unexpected token {token!r}")
        else:
            return token

    def eval(self, locals: Optional[Dict[str, Any]] = None, globals: Optional[Dict[str, Any]] = None):
        if locals is None:
            locals: Dict[str, Any] = {}
        if globals is None:
            globals: Dict[str, Any] = DEFAULT_GLOBALS
        values: List[Any] = []
        for t in self.tokens:
            if isinstance(t, FixedSizeCollection):
                args = [
                    self.get_value(arg, locals, globals) for arg in values[-t.size:]
                ]
                values = values[:-t.size] + [t.container_type(args)]
            elif isinstance(t, OperatorToken):
                args = []
                for expand, v in zip(t.op.expand, values[-t.op.arity:]):
                    if expand:
                        args.append(self.get_value(v, locals, globals))
                    else:
                        args.append(v)
                values = values[:-t.op.arity] + [t.op.execute(*args)]
            else:
                values.append(t)
        if len(values) != 1:
            raise RuntimeError(f"Unexpected extra tokens: {values[:-1]}")
        elif isinstance(values[0], IdentifierToken):
            values[0] = self.get_value(values[0], locals, globals)
        return values[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(rpn={self.tokens!r})"


def parse(expression_str: str) -> Expression:
    return Expression(infix_to_rpn(tokenize(expression_str)))
