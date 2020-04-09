from abc import abstractmethod, ABC
from typing import Iterator, List, Optional

from .printer import Back, Fore, Printer
from .search import IterativeTighteningSearch
from .bounds import Range, Bounded
from .tree import TreeNode


class Edit(Bounded):
    def __init__(self,
                 from_node,
                 to_node=None,
                 constant_cost: Optional[int] = 0,
                 cost_upper_bound: Optional[int] = None):
        self.from_node: TreeNode = from_node
        self.to_node: TreeNode = to_node
        self._constant_cost = constant_cost
        self._cost_upper_bound = cost_upper_bound
        self._valid: bool = True
        self.initial_cost = self.cost()

    @property
    def valid(self) -> bool:
        return self._valid

    @valid.setter
    def valid(self, is_valid: bool):
        self._valid = is_valid

    def tighten_bounds(self) -> bool:
        return False

    @abstractmethod
    def print(self, printer: Printer):
        pass

    def __lt__(self, other):
        return self.cost() < other.cost()

    def cost(self) -> Range:
        lb = self._constant_cost
        if self._cost_upper_bound is None:
            if self.to_node is None:
                ub = self.initial_cost.upper_bound
            else:
                ub = self.from_node.total_size + self.to_node.total_size + 1
        else:
            ub = self._cost_upper_bound
        return Range(lb, ub)

    def bounds(self):
        return self.cost()


class CompoundEdit(Edit, ABC):
    @abstractmethod
    def edits(self) -> Iterator[Edit]:
        pass

    def print(self, printer: Printer):
        for edit in self.edits():
            edit.print(printer)

    def __iter__(self) -> Iterator[Edit]:
        return self.edits()


class PossibleEdits(CompoundEdit):
    def __init__(
            self,
            from_node: TreeNode,
            to_node: TreeNode,
            edits: Iterator[Edit] = (),
            initial_cost: Optional[Range] = None
    ):
        if initial_cost is not None:
            self.initial_cost = initial_cost
        self._search: IterativeTighteningSearch[Edit] = IterativeTighteningSearch(
            possibilities=edits,
            initial_bounds=initial_cost
        )
        while not self._search.bounds().finite:
            self._search.tighten_bounds()
        super().__init__(from_node=from_node, to_node=to_node)

    @property
    def valid(self) -> bool:
        if not super().valid:
            return False
        while self._search.best_match is not None and not self._search.best_match.valid:
            self._search.remove_best()
        is_valid = self._search.best_match is not None
        if not is_valid:
            self.valid = False
        return is_valid

    @valid.setter
    def valid(self, is_valid: bool):
        self._valid = is_valid

    def best_possibility(self) -> Edit:
        return self._search.best_match

    def edits(self) -> Iterator[Edit]:
        best = self.best_possibility()
        if best is not None:
            yield best

    def tighten_bounds(self) -> bool:
        tightened = self._search.tighten_bounds()
        # Calling self.valid checks whether our best match is invalid
        self.valid
        return tightened

    def cost(self) -> Range:
        return self._search.bounds()


class Match(Edit):
    def __init__(self, match_from: TreeNode, match_to: TreeNode, cost: int):
        super().__init__(
            from_node=match_from,
            to_node=match_to,
            constant_cost=cost,
            cost_upper_bound=cost
        )

    def print(self, printer: Printer):
        if self.cost() > Range(0, 0):
            with printer.bright().background(Back.RED).color(Fore.WHITE):
                with printer.strike():
                    self.from_node.print(printer)
            with printer.color(Fore.CYAN):
                printer.write(' -> ')
            with printer.bright().background(Back.GREEN).color(Fore.WHITE):
                with printer.under_plus():
                    self.to_node.print(printer)
        else:
            self.from_node.print(printer)

    def __repr__(self):
        return f"{self.__class__.__name__}(match_from={self.from_node!r}, match_to={self.to_node!r}, cost={self.cost().lower_bound!r})"


class Replace(Edit):
    def __init__(self, to_replace: TreeNode, replace_with: TreeNode):
        cost = max(to_replace.total_size, replace_with.total_size) + 1
        super().__init__(
            from_node=to_replace,
            to_node=replace_with,
            constant_cost=cost,
            cost_upper_bound=cost
        )

    def print(self, printer: Printer):
        self.from_node.print(printer)
        if self.cost().upper_bound > 0:
            with printer.bright().color(Fore.WHITE).background(Back.RED):
                self.from_node.print(printer)
            with printer.color(Fore.CYAN):
                printer.write(' -> ')
            with printer.bright().color(Fore.WHITE).background(Back.GREEN):
                self.to_node.print(printer)
        else:
            self.from_node.print(printer)

    def __repr__(self):
        return f"{self.__class__.__name__}(to_replace={self.from_node!r}, replace_with={self.to_node!r})"


class Remove(Edit):
    def __init__(self, to_remove: TreeNode, remove_from: TreeNode):
        super().__init__(
            from_node=to_remove,
            to_node=remove_from,
            constant_cost=to_remove.total_size + 1,
            cost_upper_bound=to_remove.total_size + 1
        )

    def print(self, printer: Printer):
        with printer.bright():
            with printer.background(Back.RED):
                with printer.color(Fore.WHITE):
                    if not printer.ansi_color:
                        printer.write('~~~~')
                        self.from_node.print(printer)
                        printer.write('~~~~')
                    else:
                        with printer.strike():
                            self.from_node.print(printer)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.from_node!r}, remove_from={self.to_node!r})"


class Insert(Edit):
    def __init__(self, to_insert: TreeNode, insert_into: TreeNode):
        super().__init__(
            from_node=to_insert,
            to_node=insert_into,
            constant_cost=to_insert.total_size + 1,
            cost_upper_bound=to_insert.total_size + 1
        )

    @property
    def insert_into(self) -> TreeNode:
        return self.to_node

    @property
    def to_insert(self) -> TreeNode:
        return self.from_node

    def print(self, printer: Printer):
        with printer.bright().background(Back.GREEN).color(Fore.WHITE):
            if not printer.ansi_color:
                printer.write('++++')
                self.to_insert.print(printer)
                printer.write('++++')
            else:
                with printer.under_plus():
                    self.to_insert.print(printer)

    def __repr__(self):
        return f"{self.__class__.__name__}(to_insert={self.to_insert!r}, insert_into={self.insert_into!r})"


class EditSequence(CompoundEdit):
    def __init__(self, from_node: TreeNode, to_node: Optional[TreeNode], edits: Iterator[Edit]):
        self._edit_iter: Iterator[Edit] = edits
        self._sub_edits: List[Edit] = []
        cost_upper_bound = from_node.total_size + 1
        if to_node is not None:
            cost_upper_bound += to_node.total_size
        self._cost = None
        super().__init__(from_node=from_node,
                         to_node=to_node,
                         cost_upper_bound=cost_upper_bound)

    @property
    def valid(self) -> bool:
        if not super().valid:
            return False
        is_valid = True
        if self._edit_iter is None:
            for e in self._sub_edits:
                if not e.valid:
                    is_valid = False
                    break
        if not is_valid:
            self.valid = False
        return is_valid

    @valid.setter
    def valid(self, is_valid: bool):
        self._valid = is_valid

    def print(self, printer: Printer):
        for sub_edit in self.sub_edits:
            sub_edit.print(printer)

    @property
    def sub_edits(self) -> List[Edit]:
        while self._edit_iter is not None and self.tighten_bounds():
            pass
        return self._sub_edits

    def edits(self) -> Iterator[Edit]:
        yield from iter(self.sub_edits)

    def _is_tightened(self, starting_bounds: Range) -> bool:
        return not self.valid or self.cost().lower_bound > starting_bounds.lower_bound or \
            self.cost().upper_bound < starting_bounds.upper_bound

    def tighten_bounds(self) -> bool:
        if not self.valid:
            return False
        starting_bounds: Range = self.cost()
        while True:
            if self._edit_iter is not None:
                try:
                    next_edit: Edit = next(self._edit_iter)
                    if isinstance(next_edit, EditSequence):
                        self._sub_edits.extend(next_edit.sub_edits)
                    else:
                        self._sub_edits.append(next_edit)
                except StopIteration:
                    self._edit_iter = None
                if self._is_tightened(starting_bounds):
                    return True
            tightened = False
            for child in self._sub_edits:
                if child.tighten_bounds():
                    self._cost = None
                    if not child.valid:
                        self.valid = False
                        return True
                    tightened = True
                    new_cost = self.cost()
                    # assert new_cost.lower_bound >= starting_bounds.lower_bound
                    # assert new_cost.upper_bound <= starting_bounds.upper_bound
                    if new_cost.lower_bound > starting_bounds.lower_bound or \
                            new_cost.upper_bound < starting_bounds.upper_bound:
                        return True
                else:
                    assert not child.valid or child.cost().definitive()
            if not tightened and self._edit_iter is None:
                return self._is_tightened(starting_bounds)

    def cost(self) -> Range:
        if not self.valid:
            self._cost = Range()
        if self._cost is not None:
            return self._cost
        elif self._edit_iter is None:
            # We've expanded all of the sub-edits, so calculate the bounds explicitly:
            total_cost = sum(e.cost() for e in self._sub_edits)
            if total_cost.definitive():
                self._cost = total_cost
        else:
            # We have not yet expanded all of the sub-edits
            total_cost = Range(0, self._cost_upper_bound)
            for e in self._sub_edits:
                total_cost.lower_bound += e.cost().lower_bound
                total_cost.upper_bound -= e.initial_cost.upper_bound - e.cost().upper_bound
        return total_cost

    def __len__(self):
        return len(self.sub_edits)

    def __repr__(self):
        return f"{self.__class__.__name__}(*{self.sub_edits!r})"
