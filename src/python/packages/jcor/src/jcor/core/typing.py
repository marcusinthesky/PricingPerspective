"""Shared type vocabulary — arrays, structural laws, and legacy ``DMat`` brands.

Position
--------
rank 0 · importable by every stage

Every stage imports its array aliases from **here**, not from ``jaxtyping``
directly, so the vocabulary can be swapped in one place.

Why jaxtyping is imported at runtime in this module
---------------------------------------------------
Measured, not assumed. ``flake8-type-checking`` in ``strict`` mode would push
``from jaxtyping import Array, Float`` under ``if TYPE_CHECKING:``. With
``from __future__ import annotations`` that makes ``Float`` and ``Array``
*unresolvable names* at runtime — and beartype then **silently skips** the
annotation rather than failing. Probed under the jaxtyping import hook:

.. code-block:: text

    TYPE_CHECKING-guarded  Float[Array, "n n"]  <- shape NOT enforced
    runtime-imported       Float[Array, "n n"]  <- TypeCheckError, as intended

Since pyrefly does not verify shape algebra either (``Float[Array, "k d"]``
passed into a ``Float[Array, "n n"]`` parameter is zero static errors), a guarded
import means the shape strings are decoration in **both** planes. The import here
is therefore deliberately at runtime, and re-exported so no stage has to repeat
the reasoning.

Structural law migration
------------------------
``LawT`` is the covariant parameter for the orthogonal read-only protocols in
:mod:`jcor.core.axioms`. ``Ax`` remains separately bound to the temporary
nominal ``DMat`` marker chain until t55.3 migrates matrix provenance and t55.5
removes that compatibility surface; the two parameters must not be conflated.

Two rules for annotations, from the t46 typing probe
----------------------------------------------------
1. **Shape on the field, axiom on the parameter — never nested.**
   ``Float[SomeNewType, "n n"]`` type-checks and then raises
   ``TypeError: issubclass() arg 1 must be a class`` the moment the annotation is
   evaluated. Write ``DMat[Metric]`` with ``values: Float[Array, "n n"]``.
2. ``Static[T]`` marks an argument destined for ``jit(static_argnames=...)``.
   It is an ``Annotated`` alias: erased for type checking, visible to a reader
   and to any tooling that inspects ``__metadata__``.

Why every pytree field carries a leading ``*batch``
---------------------------------------------------
Also measured. ``jax.tree_util.tree_unflatten`` rebuilds a registered dataclass
by **calling** ``__init__``, and ``vmap`` unflattens its output with the batched
leaves. Under the import hook that call is type-checked, so a field annotated
``Float[Array, "n n"]`` makes ``vmap`` over any function returning a ``DMat``
fail with ``f64[2,3,3] is not Float[Array, 'n n']`` — at the *return*, long after
the per-cell construction that was correctly shaped. ``"*batch n n"`` keeps the
square constraint (``(2, 3, 4)`` is still rejected) while admitting the axis a
transform prepends.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import (
    Annotated,
    Final,
    Generic,
    ParamSpec,
    Protocol,
    SupportsIndex,
    TypeVar,
)

import jax
from jaxtyping import (
    Array,
    ArrayLike,
    Bool,
    Float,
    Int,
    Num,
    Real,
    Scalar,
    ScalarLike,
    UInt32,
)

from jcor.core.axioms import Law, Premetric

__all__ = [
    "STATIC",
    "Array",
    "ArrayLike",
    "Ax",
    "Bool",
    "DMat",
    "Float",
    "Int",
    "LawT",
    "LawfulCallable",
    "Num",
    "PRNGKey",
    "Params",
    "Real",
    "ResultT",
    "Scalar",
    "ScalarLike",
    "Static",
    "as_index",
]

#: Marker payload carried by :data:`Static`. Compared by identity, never parsed.
STATIC: Final = "jcor.static"


def as_index(value: object) -> int | None:
    """Return ``value`` as a Python ``int``, or ``None`` if it is not integral.

    Replaces the ``isinstance(value, (int, np.integer))`` test that eager doors
    across ``inference``, ``model`` and ``association`` each spelled for
    themselves before t65. Anything implementing ``__index__`` qualifies — a
    Python ``int``, a NumPy integer, a 0-d integer :class:`jax.Array` — which is
    both wider and more honest than naming one array library's scalar types.

    ``bool`` is rejected first, because it satisfies ``__index__`` and is never
    a meaningful count: ``n_folds=True`` should fail, not mean one.
    """
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        return None
    try:
        return operator.index(value)
    except TypeError:  # pragma: no cover — an object with a lying __index__
        return None


type Static[T] = Annotated[T, STATIC]
"""Mark an argument as a JIT-static (``jit(static_argnames=...)``) value.

``Static[int]`` *is* ``int`` to a type checker; the annotation records the
calling convention that ``generators/fields.py`` and the kernel modules already
rely on informally in prose.
"""

type PRNGKey = UInt32[Array, "2"]
"""A legacy JAX PRNG key as produced by :func:`jax.random.PRNGKey`.

The dtype and two-word key shape are part of the contract, rather than the
former unqualified ``jax.Array`` alias. Numerical cores accept this key; eager
adapters alone turn an integer seed into one.
"""


#: Covariant structural-law parameter for typed strategies and future
#: multi-axis mathematical containers. Unlike :data:`Ax`, it is not restricted
#: to the temporary nominal marker chain.
LawT = TypeVar("LawT", bound=Law, covariant=True)  # noqa: PLC0105

#: Exact callable parameter list propagated by :class:`LawfulCallable`.
Params = ParamSpec("Params")

#: Covariant callable result propagated by :class:`LawfulCallable`.
ResultT = TypeVar("ResultT", covariant=True)  # noqa: PLC0105


class LawfulCallable(Protocol[Params, ResultT, LawT]):
    """A callable retaining its full signature, result, and read-only law type."""

    @property
    def law(self) -> LawT:
        """Mathematical law declared for values produced by this callable."""
        ...

    def __call__(
        self,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> ResultT:
        """Evaluate the callable without erasing its exact signature."""
        ...


#: Legacy axiom brand of a :class:`DMat`. **Covariant**: a narrower brand flows into a
#: wider requirement (``DMat[Metric]`` satisfies ``DMat[Semimetric]``), while the
#: reverse is a pyrefly error.
#:
#: The explicit ``TypeVar`` is required, not stylistic. ``Ax`` is a phantom
#: parameter — it appears in no field — so PEP 695 (``class DMat[Ax: Premetric]``)
#: has nothing to infer variance from and settles on **invariant**. Measured
#: under pyrefly strict: the PEP 695 spelling rejects ``DMat[Metric]`` where
#: ``DMat[Semimetric]`` is required, which is the lattice's whole purpose.
#: `PLC0105` would rename this ``Ax_co``; the t46 contract fixes the public name
#: as ``Ax`` and the suffix would leak the mechanism into the vocabulary.
Ax = TypeVar("Ax", bound=Premetric, covariant=True)  # noqa: PLC0105


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DMat(Generic[Ax]):  # noqa: UP046  # PEP 695 loses covariance; see `Ax`
    """A square dissimilarity matrix branded with the axioms it satisfies.

    ``Ax`` is a *phantom* parameter: it appears in no field, costs nothing at
    runtime, and never reaches a JAX transform. Registered as a pytree, so
    ``tree_leaves(DMat(jnp.eye(4))) == [(4, 4)]`` and ``jit``/``vmap`` see only
    the wrapped array.

    Attributes:
        values: The dissimilarity matrix, shape ``(n, n)`` — or ``(*batch, n, n)``
            once a transform has batched it; see the module docstring. The shape
            lives on this field; the axiom lives on the class parameter. Never
            nest them.

    Examples:
        >>> from jcor.core.axioms import Metric
        >>> d: DMat[Metric] = DMat(values=jnp.eye(4))
        >>> d.values.shape
        (4, 4)

    """

    values: Float[Array, "*batch n n"]  # noqa: F722  # jaxtyping shape, not a forward ref
