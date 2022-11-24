from __future__ import annotations
import re
import inspect
import logging
import os
import sys
from collections.abc import Callable
from enum import Enum
from threading import RLock
from types import FunctionType
from typing import Any, Generic, List, Optional, TypeVar, cast, Dict, Tuple, Type, Union
from inspect import _empty

InjectedTypeDef = TypeVar("InjectedTypeDef")
# copied from python dependency injector
if sys.version_info < (3, 7):
    from typing import GenericMeta
else:

    class GenericMeta(type):
        """Delcaring a missing type"""


class ClassMetaInfo(GenericMeta):
    """
    Class item meta
    """

    def __getitem__(cls, item):
        """
        Must have method, or the metaclass won't be able to apply
        """
        # Spike for Python 3.6
        if isinstance(item, tuple):
            return cls(*item)
        return cls(item)


class Injected(Generic[InjectedTypeDef], metaclass=ClassMetaInfo):
    """
    Injected meta for functions to mark a argument value need to be injected.
    CAUTION: be aware of your argument name, the framework will try to lookup bean by argument name first.

    Parameters
    ----
    type: type, require
        The type of the argument
    """

    def __init__(self, real_type: Type[Any], bean_name: str = ""):
        self.real_type = real_type
        self.bean_name = bean_name


def is_debug() -> bool:
    """
    Check if the logger level is debug

    Returns
    ---
    bool
        True if the log level is debug, False otherwise
    """
    return logger.root.level == logging.DEBUG


def is_function(item: Union[Callable, type]) -> bool:
    """
    Check if item is Callable but not a class

    Parameters
    ----
    item: Any
        The item to be checked

    Returns
    bool
        True if the item is Callable and not a class
    """
    if isinstance(item, type):
        return callable(item)
    return isinstance(item, FunctionType)


def get_method_delcared_class(method: Callable) -> Tuple[bool, Optional[ClassFqn]]:
    """
    Get the declared class for the method

    Parameters
    ----
    method: Callable
        The method to be inspect

    Returns
    ----
    Tuple[bool, Tuple[ClassFqn]]
        First bool element to tell whether the method could be inside class
        Second ClassFqn to describe the class full qualified name
    """
    is_method = is_function(method)
    if is_method:
        fqn = method.__qualname__
        parts = fqn.split(".")
        logger.debug(f"====Method {fqn}")
        if len(parts) == 1:
            return False, None
        container_path = ".".join(parts[0:-1])
        module_name = method.__module__
        logger.debug(f"====Method {fqn} container path {module_name}.{container_path}")
        return True, ClassFqn(module_name, container_path)
    logger.debug(f"{method.__name__} {method} type={method} is not a method ?")
    return False, None


logger = logging
logger.basicConfig(
    level=int(os.getenv("PY_LOG_LEVEL", "20")),
    format="%(name)s - %(levelname)s - %(pathname)s#%(lineno)d - %(message)s",
)

INTERNAL_CALL_FLAG = "__di_module_call_identifier__"
METHOD_ANNOTATION_META_INJECTED = "__di_method_annotation_meta_injected__"
METHOD_ANNOTATION_META_BEAN = "__di_method_annotation_meta_bean__"
METHOD_ANNOTATION_REAL_METHOD_REFERENCE = "__di_method_annotation_real_method_ref__"


class ClassFqn:  # pylint: disable=too-few-public-methods
    """
    Class's full qualified name definition

    Attributes
    ----
    module: str
        Module name which contains the class
    class_name: str
        Class name
    fqn: str
        {module}.{class_name}
    """

    def __init__(self, module: str, class_name: str):
        self.module = module
        self.class_name = class_name
        self.fqn = f"{module}.{class_name}"

    def __str__(self):
        return f"ClassFqn[{self.fqn}]"


class Scope(Enum):
    """Scope of the bean, used by @bean() annotation, for example
    @bean(Scope.SINGLETON)

    SINGLETON: mark the bean will have only one instance
    PROTOTYPE: mark the bean will give you a new instance when you ask for it
    """

    SINGLETON = 1
    PROTOTYPE = 2


class InjectedParam:
    def __init__(self, param_name: str, bean_name: str) -> None:
        self.param_name = param_name
        self.bean_name = bean_name


class BaseBeanDef:
    """A basic bean definition.

    Attributes
    ----
    name: str
        Bean's alias
    type: type
        Bean's type
    """

    def __init__(self, name: str, bean_type: Type[Any] | ClassFqn):
        self.name = name
        self.type = bean_type
        # ref by ClassFqn
        self.shadow_type: Optional[ClassFqn] = bean_type if isinstance(bean_type, ClassFqn) else None

    def __eq__(self, another) -> bool:
        return (
            isinstance(another, BaseBeanDef)
            and another.name == self.name
            and another.type == self.type
            and another.shadow_type == self.shadow_type
        )

    def __hash__(self) -> int:
        return hash(self.__class__) + hash(self.name) + hash(self.type) + hash(self.shadow_type)

    def __str__(self) -> str:
        return f"BaseBeanDef(name={self.name}, type={self.type})"


class Bean(BaseBeanDef):
    """Singlton bean definition.

    Attributes
    ----
    instance: Any
        Singlton instance of the bean
    base_bean_def: BaseBeanDef
        A BaseBeanDef instance of the bean
    """

    def __init__(self, name: str, t: type, instance: Any, meta: BeanProviderMeta):
        super().__init__(name, t)
        self.instance = instance
        self.base_bean_def = BaseBeanDef(name, t)
        self.meta = meta

    def __eq__(self, another) -> bool:
        return (
            isinstance(another, Bean)
            and another.name == self.name
            and another.type == self.type
            and another.instance == self.instance
        )

    def __hash__(self) -> int:
        return hash(self.__class__) + super().__hash__() + hash(self.meta)


class PrototypeBean(Bean):
    """PrototypeBean will give you new instance at each time you ask for it.

    Attributes
    ----
    meta: BeanProviderMeta
        The bean provider description for this bean
    """

    def __init__(self, name: str, bean_type: Type[Any], meta: BeanProviderMeta):
        super().__init__(name, bean_type, meta, meta)

    def __eq__(self, another) -> bool:
        return (
            isinstance(another, PrototypeBean)
            and another.name == self.name
            and another.type == self.type
            and another.instance == self.instance
        )

    def __hash__(self) -> int:
        return hash(self.__class__) + super().__hash__() + hash(self.meta)

    def __str__(self):
        return f"PrototypeBean(name={self.name}, type={self.type}, meta={self.meta})"


class ScopedBeanDef(BaseBeanDef):
    """A bean definition which contains scope specificied.

    Attributes
    ----
    scope: Scope
        The scope of this bean
    base_bean_def: BaseBeanDef
        BaseBeanDef for this bean
    """

    def __init__(self, name: str, t: Type[Any], scope: Scope):
        super().__init__(name, t)
        self.scope = scope
        self.base_bean_def = BaseBeanDef(name, t)

    def __eq__(self, another) -> bool:
        basic_compare = isinstance(another, ScopedBeanDef) and another.type == self.type and another.scope == self.scope
        # for prototype bean, we do not need to care about its name
        if self.scope == Scope.PROTOTYPE:
            return basic_compare
        return basic_compare and another.name == self.name

    def __hash__(self) -> int:
        basic_hash = hash(self.__class__) + hash(self.type) + hash(self.scope)
        # for prototype bean, we do not need to care about its name
        if self.scope == Scope.PROTOTYPE:
            return basic_hash
        return basic_hash + hash(self.name)

    def __str__(self):
        return f"ScopedBeanDef(name={self.name}, type={self.type}, scope={self.scope})"


class ScopedBeanDefWithThinDeps(ScopedBeanDef):
    """Bean definition with scope and its dependencies.

    Attributes
    ----
    depends_on: List[BaseBeanDef]
        Beans depended by this bean.
    depended_by: Optional[ScopedBeanDefWithThinDeps]
        The bean which requires this bean before its initialization.
    method_hash: int
        Hash value for the factory_method.
    class_method_flag: bool
        Flag whether the factory method is a class method
    class_reference: Optional[ClassFqn]
        The class description for the class which contains the factory method.
    """

    def __init__(
        self,
        name: str,
        t: Type[Any],
        scope: Scope,
        depends_on: List[BaseBeanDef],
        depended_by: Optional[ScopedBeanDefWithThinDeps] = None,
        method_hash: int = 0,
        class_method_flag: bool = False,
        class_reference: Optional[ClassFqn] = None,
    ):  # pylint: disable=too-many-arguments
        super().__init__(name, t, scope)
        self.depends_on = depends_on
        self.depended_by = depended_by
        self.scoped_bean_def = ScopedBeanDef(name, t, scope)
        self.method_hash = method_hash
        self.class_method_flag = class_method_flag
        self.class_reference = class_reference

    def __eq__(self, another) -> bool:
        basic_eq = (
            isinstance(another, ScopedBeanDefWithThinDeps)
            and another.scoped_bean_def == self.scoped_bean_def
            and another.depends_on == self.depends_on
            and another.class_method_flag == self.class_method_flag
            and another.class_reference == self.class_reference
        )
        return basic_eq

    def __hash__(self) -> int:
        basic_hash = (
            hash(self.__class__)
            + hash(self.type)
            + hash(self.scope)
            + self.method_hash
            + hash(self.class_method_flag)
            + hash(self.class_reference)
        )
        if self.scope == Scope.PROTOTYPE:
            return basic_hash
        return basic_hash + hash(self.name)

    def __str__(self):
        depends_on_str = "".join(["[" + str(x) + "]" for x in self.depends_on]) if len(self.depends_on) > 0 else "None"
        return (
            f"name={self.name}, type={self.type}, scope={self.scope}, "
            + f"depended by {self.depended_by}, depends on {depends_on_str}"
        )


class BeanDependencyChainBuilder:
    """Dependency chain between beans. It will collect bean's definition, and
    analyze the init order for each bean.

    Attributes
    ----
    raw_def: List[ScopedBeanDefWithThinDeps]
        The very original bean definitions
    raw_defs_map: dict[BaseBeanDef, ScopedBeanDefWithThinDeps]
        A map which uses a BaseBeanDef as key, and ScopedBeanDefWithThinDeps as its value
    bare_type_and_raw_defs_map: dict[BaseBeanDef, ScopedBeanDef]
        A map which will be used for looking up its ScopedBeanDef via BaseBeanDef(contains no scope).
    init_order_map: dict[BaseBeanDef, init]
        A map holding beans' initialization order.
    """

    def __init__(self) -> None:
        self.raw_defs: List[ScopedBeanDefWithThinDeps] = []
        self.raw_defs_map: Dict[BaseBeanDef, ScopedBeanDefWithThinDeps] = {}
        self.bare_type_and_raw_defs_map: Dict[BaseBeanDef, ScopedBeanDef] = {}
        self.init_order_map: dict[BaseBeanDef, int] = {}

    def get_bean_creation_order(self) -> List[BaseBeanDef]:
        """Get the init order for beans, the BaseBeanDef at index 0 should be
        inited at first place.

        Returns
        ----
        List[BaseBeanDef]
            Beans' initialization order specificied with a list.
        """
        keys = list(self.init_order_map.keys())
        values = [v for (_, v) in self.init_order_map.items()]
        keys = [x for _, x in sorted(zip(values, keys), key=lambda pair: pair[0])]
        sorted(keys, key=lambda x: int(self.init_order_map[x]))
        if logger.root.level == logging.DEBUG:
            for key in keys:
                logger.debug(f"Will init {key} at order {self.init_order_map[key]}.")
        return keys

    def register_bean_def(self, bean_def: ScopedBeanDefWithThinDeps) -> BeanDependencyChainBuilder:
        """Registering bean's definition.

        Parameters
        ----
        bean_def: ScopedBeanDefWithThinDeps
            The bean's definition

        Returns
        BeanDependencyChainBuilder
        """
        scoped_bean_def = bean_def.scoped_bean_def
        logger.debug(
            "Registering bean definition {bean_def} hash={hash(bean_def)} scoped-bean-def-hash={scoped_bean_def}"
        )
        self.raw_defs.append(bean_def)
        current_value = self.raw_defs_map.setdefault(scoped_bean_def, bean_def)
        # use protype scope if there's conflicts happened
        if current_value != bean_def and bean_def.scope == Scope.PROTOTYPE:
            logger.debug(f"Updating value in raw_defs_map {scoped_bean_def} to {bean_def}")
            self.raw_defs_map[scoped_bean_def] = bean_def

        bare_type_raw_def = self.bare_type_and_raw_defs_map.setdefault(bean_def.base_bean_def, scoped_bean_def)
        # use protype scope if there's conflicts happened
        if bare_type_raw_def != bean_def and bean_def.scope == Scope.PROTOTYPE:
            logger.debug(f"Updating value in bare_type_and_raw_defs_map {bean_def.type} to {bean_def}")
            self.bare_type_and_raw_defs_map[bean_def.base_bean_def] = scoped_bean_def
        return self

    def increase_type_init_order(
        self, bean_def: BaseBeanDef, qty: int, update_depdencies: bool = True, path: Optional[List[BaseBeanDef]] = None
    ):
        """Increase bean's init order, can be used as decrease init order by
        passing negative order.

        Parameters
        ----
        bean_def: BaseBeanDef, required
            The bean which will be affected
        order: int, required
            The order value affected. Positive value for marking the bean will be inited latter.
        update_depdencies: bool, optional, defualt is True
            Whether to update bean's dependencies as well.
        path: Optional[List[BaseBeanDef]]
            The dependencies path of current BaseBeanDef.

        Raises
        ----
        Exception
            If circular dependency found, or the BaseBeanDef could not be found,
            or bean's dependencies could not be found.
        """
        if path is None:
            path = []
        old_order = self.init_order_map[bean_def] if bean_def in self.init_order_map else 0
        final_order = int(qty) + int(old_order)
        self.init_order_map[bean_def] = final_order
        path_str = " > ".join([str(x) for x in path])
        logger.debug(f"Bean {bean_def} init order is {final_order},  dep path: {path_str}")
        if not update_depdencies:
            return
        self.prepare_dependencies(bean_def, abs(qty), path)

    def logging_raw_def_maps(self):
        if not is_debug():
            return
        logger.debug("Will print registered bean")
        keys = list(self.raw_defs_map.keys())
        for key in keys:
            definition = self.raw_defs_map[key]
            logger.debug(f"type {key} = {definition} hash={hash(key)}")

    def logging_bean_init_order(self):
        if not is_debug():
            return
        definitions = self.get_bean_creation_order()
        order = 0
        for item in definitions:
            logger.debug(f"[{order}] the init order for {item}")
            order = order + 1

    def prepare_dependencies(
        self,
        base_bean_def_ref: BaseBeanDef,
        base_amount: int = 1,
        existing_call_path: Optional[List[BaseBeanDef]] = None,
    ):
        """
        Prepare the dependencies for base_bean_def, so its dependencies can be inited in proper order.

        base_bean_def: BaseBeanDef
            The start point for dependencies
        base_amount: int
            The init order affected
        existing_call_path: Optional[List[BaseBeanDef]]
            The dependencies path, for detecting circular dependencies
        """
        if existing_call_path is None:
            existing_call_path = []
        # todo: may find a better way to do it?
        if existing_call_path.count(base_bean_def_ref) > 1:
            path_temp = list(existing_call_path)
            path_temp.append(base_bean_def_ref)
            dep_list = [f"{x}" for x in path_temp]
            dpe_list_str = " -> ".join(dep_list)
            raise Exception(f"Circular dependencies found: {dpe_list_str}")
        if base_bean_def_ref not in self.raw_defs_map:
            raise Exception(
                f"Could not find {base_bean_def_ref} hash={hash(base_bean_def_ref)} from bean definition map"
            )
        bean_def = self.raw_defs_map[base_bean_def_ref]
        # increase init order for bean which needs this type
        if bean_def.depended_by is not None:
            tmp_path = [base_bean_def_ref]
            tmp_path.extend(existing_call_path)
            self.increase_type_init_order(bean_def, base_amount * -1, False, tmp_path)
            self.increase_type_init_order(bean_def.depended_by, base_amount, True, tmp_path)
        depends_on = bean_def.depends_on
        if depends_on is None or len(depends_on) == 0:
            return
        bare_types = self.bare_type_and_raw_defs_map.keys()
        for dep_bean_def in depends_on:
            real_type = dep_bean_def.type
            if isinstance(real_type, ClassFqn):
                old_type = dep_bean_def
                fetched_type = BeanFactory.get_class(cast(ClassFqn, dep_bean_def.type))
                ref_type = dep_bean_def.type
                dep_bean_def = BaseBeanDef(BeanFactory.snake_case(fetched_type.__name__), fetched_type)
                dep_bean_def.shadow_type = cast(ClassFqn, ref_type)
                real_type = fetched_type
                logger.debug(f"Corrected depends on from {old_type} to {dep_bean_def}")
            if dep_bean_def not in self.bare_type_and_raw_defs_map:
                # trying to find any subclass can work
                real_type_ref = cast(Type, real_type)
                filtered_data = filter(
                    lambda x: isinstance(x.type, type)
                    and issubclass(x.type, real_type_ref),  # pylint: disable=cell-var-from-loop
                    bare_types,
                )
                found_types = list(filtered_data)
                if len(found_types) == 0:
                    raise Exception(
                        f"Failed to find bean definition of {bean_def.type}"
                        + f" which is depended by {base_bean_def_ref.type}"
                    )
                # use one that will work
                # trying to find by bean name, or use matched first item
                # todo: may also check bean's type as well
                result_by_name = list(filter(lambda x: x.name == bean_def.name, found_types))
                old_type = dep_bean_def
                dep_bean_def = found_types[0] if len(result_by_name) == 0 else result_by_name[0]
                found_types_str = ", ".join([str(x) for x in found_types])
                logger.debug(f"Chosen matched type {dep_bean_def} for {old_type} from {found_types_str}")
            the_bean_def = self.bare_type_and_raw_defs_map[dep_bean_def]
            tmp_path = list(existing_call_path)
            tmp_path.append(base_bean_def_ref)
            amount = -1 * base_amount
            logger.debug(
                f"Trying to increate order for {the_bean_def} by {amount} because it is needed by"
                + f"{base_bean_def_ref}, calling path = {tmp_path}"
            )
            self.increase_type_init_order(the_bean_def, amount, True, tmp_path)

    def prepare_for_bean_creation(self) -> BeanDependencyChainBuilder:
        """Prepare beans' init order data.

        Raises
        ----
        Exception
            Any bean's definition was missing from config, it probably a BUG.
        """
        self.logging_raw_def_maps()
        all_types = self.raw_defs_map.keys()
        for current_type in all_types:
            self.increase_type_init_order(current_type, 0, False)
        for current_type in all_types:
            self.prepare_dependencies(current_type)
        self.logging_bean_init_order()
        return self


class BeanRegistry:
    """Bean's registry, can be used for looking up a bean without injection
    point declared."""

    _instance: Optional[BeanRegistry] = None
    _beans_name_map: dict[str, Bean] = {}
    _beans_type_map: dict[BaseBeanDef, List[Bean]] = {}
    _bean_lock = RLock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> BeanRegistry:
        if cls._instance is None:
            return BeanRegistry()
        return cls._instance

    def __init__(self) -> None:
        self.__beans_name_map: dict[str, Bean] = BeanRegistry._beans_name_map
        self.__beans_type_map: dict[BaseBeanDef, List[Bean]] = BeanRegistry._beans_type_map
        self.__bean_lock = BeanRegistry._bean_lock

    def register_bean(self, bean_ref: Bean):
        """Registering a bean into registry.

        Parameters
        ----
        bean: Bean
            The bean's instance.
        """
        with self.__bean_lock:
            result = self.__beans_name_map.setdefault(bean_ref.name, bean_ref)
            if result != bean_ref:
                raise Exception(f"Bean of name={bean_ref.name} type={bean_ref.type} was already existed.")
            bean_list = self.__beans_type_map.setdefault(bean_ref.base_bean_def, [])
            if len(bean_list) == 0 or bean_list.count(bean_ref) == 0:
                bean_list.append(bean_ref)

    def __get_or_create(self, bean_ref: Bean) -> Any:
        """Get a bean if it is Singlton, or create a new instance if it is
        protype.

        Parameters
        ----
        bean: Bean
            Bean's definition

        Raises
        ----
        Exception
            The bean could not be found

        Returns
        ----
        Any: The bean's instance, could be any type.
        """
        is_correct_type = isinstance(bean_ref, Bean)
        if bean_ref is None or not is_correct_type:
            raise Exception(f"Could not find a bean named {bean_ref.name}")
        if isinstance(bean_ref, PrototypeBean):
            return bean_ref.instance()
        return bean_ref.instance

    def one_by_name(self, name: str) -> Any:
        """Retrieve a bean by its name.

        Parameters
        ----
        name: str
            Bean's name

        Raises
        ----
        Exception
            If the bean's name was not found.

        Returns
        ----
        Any: The bean's instance, could be any type.
        """
        if not name in self.__beans_name_map:
            raise Exception(f"Could not find a bean named {name}")
        bean_ref = self.__beans_name_map[name]
        return self.__get_or_create(bean_ref)

    def one_by_type(self, bean_type: Type[Any]) -> Any:
        """Get bean by it type.

        Parameters
        ----
        bean_type: type
            The definition for the bean

        Raises
        ----
        Exception
            If the bean was not found or more than one instance found.

        Returns
        ----
        Any: The bean's instance, could be any type.
        """
        items = self.list_by_type(BaseBeanDef("", bean_type))
        if len(items) > 1:
            raise Exception(f"Could not determine which instance to use for {bean_type} from {items} candidates.")
        return items[0]

    def one_by_name_or_type(self, name: str, bean_type: Type[Any]) -> Any:
        """Get a bean by its name or type.

        Parameters
        ----
        name: str
            Bean's name
        bean_type: type
            Bean's type

        Raises
        ----
        Exception
            If the bean could not be found or more than one instance found.
        """
        try:
            return self.one_by_name(name)
        except:  # pylint: disable=bare-except
            return self.one_by_type(bean_type)

    def list_by_type(self, bean_def: BaseBeanDef) -> List[Any]:
        """Find a instances list of specificied bean type.

        Parameters
        ----
        bean_def: BaseBeanDef
            Beans definition

        Raises
        ----
        Exception
            If nothing was found.

        Returns
        ----
        List[Any]: A list of found instances.
        """
        items: List[Bean] = []
        if not bean_def in self.__beans_name_map:
            items = self.__get_matching_types(BeanFactory.get_class_by_bean_def(bean_def))
        else:
            items = self.__beans_type_map[bean_def]
            if len(items) == 0:
                items = self.__get_matching_types(cast(Type, bean_def.type))
        if len(items) == 0:
            raise Exception(f"Could not find a bean for type {bean_def}")
        bean_name = bean_def.name
        # trying to matching bean by name first
        result_by_name = list(filter(lambda x: self.bean_name_filter_method(x, bean_name, bean_def), items))
        items = result_by_name if len(result_by_name) > 0 else items
        items = [self.__get_or_create(x) for x in items]
        return items

    def bean_name_filter_method(self, current_bean: Bean, name: str, bean_def: BaseBeanDef) -> bool:
        return (
            current_bean.name == name
            and isinstance(current_bean.type, type)
            and issubclass(current_bean.type, BeanFactory.get_class_by_bean_def(bean_def))
        )

    def __get_matching_types(self, the_type: Type[Any]) -> List[Bean]:
        """Get a proper list of beans which matching requested type.

        Parameters
        ----
        the_type: type
            The type or super type for the result.

        Returns
        ----
        List[Bean]: a list of Bean which matching t, could be exact type or sub type.
        """
        old_type = the_type
        if isinstance(the_type, BeanProviderMeta):
            the_type = cast(BeanProviderMeta, the_type).return_type
        logger.debug(f"Trying to find matching type for {old_type}, type is {the_type}")
        keys = self.__beans_type_map.keys()
        matching_types = list(filter(lambda x: isinstance(x.type, type) and issubclass(x.type, the_type), keys))
        result: List[Bean] = []
        for item in matching_types:
            result.extend(self.__beans_type_map[item])
        return result


def will_argument_be_ignored(func: Callable, argument: str) -> bool:
    """
    Check if a argument should be ignored for injection

    Parameters
    ----
    func: Callable
        The method contains the argument
    argument: str
        The argument to be decided

    Returns
    bool
        True if the argument should be ignored, False otherwise.
    """
    method_name = func.__name__
    return method_name == "__init__" and argument == "self"


def parse_needed_injected_params(func: Callable) -> Dict[InjectedParam, Type[Any]]:
    """Parsing params should be injected by DI.

    Parameters
    ----
    func: Callable
        The method

    Returns
    ----
    dict[str, type]: A dict contains name and its type for injecting.
    """
    injected_params: Dict[InjectedParam, Type[Any]] = {}
    signature = inspect.signature(func)
    parameters = signature.parameters
    method_name = func.__name__
    logger.debug(f"function {method_name} has parameters : {parameters} ")
    for key, value in parameters.items():
        default_value = value.default
        param_type: Optional[Type] = None
        param_name = key
        bean_name = key
        if value.annotation is not None:
            param_type = value.annotation
        if isinstance(default_value, Injected):
            injected_value: Injected = cast(Injected, default_value)
            param_type = injected_value.real_type
            if len(injected_value.bean_name) > 0:
                bean_name = injected_value.bean_name
        if param_type is None:
            logger.warning(
                f"Failed to decide injected data type for {method_name}.{key}, "
                + f"annotated type is {value.annotation} and default value is {default_value}"
            )
            continue
        if will_argument_be_ignored(func, key) or param_type is _empty:
            logger.debug(f"Ignoring argument {key} of method {method_name}")
            continue
        # logger.debug("*** {}.{} can be injected with type={}".format(method_name, k, param_type))
        injected_params.setdefault(InjectedParam(param_name, bean_name), param_type)
    return injected_params


class BeanProviderMeta:
    """Bean's provider meta data. Will be used for analyze init order latter.

    Attributes
    ----
    func: Callable
        The factory method reference for the bean.
    bean_name: str
        The bean's name.
    return_type: type
        The bean's type
    scope: Scope
        Bean's scope, SINGLETON or PROTOTYPE.
    init_params_need_injection: Optional[Dict[str, Type[Any]]]
        Params to be injected for beans init method
    class_method_flag: bool
        Whether this bean's factory method is a class field, which need to be called after the class_reference inited.
    class_reference: Optional[ClassFqn]
        The fqn of the class which contain the bean's factory method. MUST use string while the @bean annotation
        invoked before the class finished loading into memory.
    """

    def __init__(
        self,
        func: Callable,
        bean_name: str,
        return_type: Type[Any],
        scope: Scope,
        init_params_need_injection: Optional[Dict[InjectedParam, Type[Any]]] = None,
        class_method_flag: bool = False,
        class_reference: Optional[ClassFqn] = None,
    ):
        self.func = func
        self.bean_name = bean_name
        self.return_type = return_type
        self.scope = scope
        self.init_params_need_injection = init_params_need_injection
        self.class_method_flag = class_method_flag
        self.class_reference = class_reference
        self.do_register()

    def __call__(self, *args, **kwargs):
        logger.debug(
            f"Invoking {self.func.__name__} is_class_member ? {self.class_method_flag}, "
            + f"class_reference = {self.class_reference}, args type={type(args)}, kwargs type = {type(kwargs)}"
        )
        if self.class_method_flag and self.class_reference is not None:
            # set proper self for fields
            class_type = BeanFactory.get_class(self.class_reference)
            class_instance = BeanRegistry.get_instance().one_by_type(class_type)
            if args is None:
                args = []
                args.append(class_instance)
                args = tuple(args)
            if isinstance(args, Tuple):
                if len(args) == 1 and args[0] != class_instance:
                    new_args = [class_instance]
                    new_args.extend(list(args))
                    args = tuple(new_args)
                elif len(args) == 0:
                    args = tuple([class_instance])

            logger.debug(
                f"Trying to setup self reference for calling method {self.func.__name__}, "
                + f"{len(args)} args {len(kwargs.keys())} kwargs."
                + f"class_type = {class_type}, instance = {class_instance}"
            )

        return BeanFactory.inject_method(self.func)(*args, **kwargs)

    def do_register(self) -> ScopedBeanDef:
        logger.debug(f"trying to register with {self.func.__name__}")
        register_result = BeanFactory.register_bean_def_with_meta(self)
        BeanInitializer.get_instance().register_bean_def(register_result, self)
        return register_result

    def __str__(self):
        params_str = (
            "None"
            if self.init_params_need_injection is None
            else ",".join([f"{k}:{v}" for (k, v) in self.init_params_need_injection.items()])
        )
        return (
            "BeanProvider(bean_name={}, type={}, factory_method={}, scope={}, "
            + "params_init={}, is_class_member={}, class_reference={})"
        ).format(
            self.bean_name,
            self.return_type,
            self.func.__name__,
            self.scope,
            params_str,
            self.class_method_flag,
            self.class_reference,
        )


class InjectedInstructionMeta:
    """Meta data for describing a injected method.

    Attributes
    ----
    func: Callable
        The method reference to real method.
    injected_params: dict[str, type]
        Analyzed dict contains a bean name as key, and its type as value.
    bean_provider_meta: BeanProviderMeta, optional
        Just in case the @bean and @injected will be using together.
    """

    def __init__(self, func: Callable, bean_provider_meta: Optional[BeanProviderMeta] = None):
        self.func = func
        self.injected_params = parse_needed_injected_params(self.func)
        self.bean_provider_meta = bean_provider_meta

    def __call__(self, *args, **kwargs):
        method = self.func
        kw_args_str = ",".join(list(kwargs))
        logger.debug(
            f"InjectedInstructionMeta {self}.__call__ {method.__name__} {len(args)} "
            + f"positional arguments, {len(kwargs)} kwargs {kw_args_str}"
        )
        # call real method or there will be stack overflow happened.
        if INTERNAL_CALL_FLAG in kwargs:
            del kwargs[INTERNAL_CALL_FLAG]
            return self.func(*args, **kwargs)
        return BeanFactory.inject_method(self.func)(*args, **kwargs)


class BeanFactory:
    """Bean's factory, all beans should be inited by this class.

    Class Attributes
    ----
    __named_factories: dict[str, BeanProviderMeta]
        Bean's provider data map, can be accessed by bean's name.
    __bean_registry: BeanProvider
        A registry holding all beans
    __global_lock: RLock
        A lock for making sure beans won't be inited by accident.
    __bean_dep_chain: BeanDependencyChainBuilder
        Keeping beans dependencies and init order.
    """

    __named_factories: Dict[str, BeanProviderMeta] = {}
    __class_ref: Dict[str, Type[Any]] = {}
    __bean_registry = BeanRegistry()
    __global_lock = RLock()
    __bean_dep_chain = BeanDependencyChainBuilder()

    @classmethod
    def registry(cls) -> BeanRegistry:
        return cls.__bean_registry

    @classmethod
    def bean_dep_chain(cls) -> BeanDependencyChainBuilder:
        """
        The beans' dependencies chain builder

        Returns
        ---
        BeanDependencyChainBuilder
            The chain builder
        """
        return cls.__bean_dep_chain

    @classmethod
    def get_class(cls, class_path: ClassFqn) -> Type[Any]:
        """
        Get the real class for class path(meta)

        Parameters
        ----
        class_path: ClassFqn
            The class fqn

        Returns
        ----
        Type[Any]
            The retrieved class.
        """
        if class_path.fqn in cls.__class_ref:
            return cls.__class_ref[class_path.fqn]
        module = sys.modules[class_path.module]
        return getattr(module, class_path.class_name)

    @classmethod
    def get_class_by_bean_def(cls, bean_def: BaseBeanDef) -> Type[Any]:
        """
        Get class type by its bean definition

        Parameters
        ----
        bean_def: BaseBeanDef
            The bean's definition

        Returns
        Type[Any]
            The type for the bean
        """
        if isinstance(bean_def.type, ClassFqn):
            return cls.get_class(cast(ClassFqn, bean_def.type))
        return cast(Type[Any], bean_def.type)

    @classmethod
    def register_bean_def(
        cls,
        name: str,
        bean_type: Type[Any],
        factory_method: Callable,
        scope: Scope = Scope.SINGLETON,
        init_params_need_injection: Optional[Dict[InjectedParam, Type[Any]]] = None,
    ) -> ScopedBeanDef:
        """Registering bean's definition.

        Parameters
        ----
        name: str
            Bean's type
        bean_type: type
            Bean's type
        factory_method: Callable
            Factory method for this bean
        scope: Scope
            SINGLETON or PROTOTYPE

        Raises
        ----
        Exception
            If the bean duplicated.
        """
        logger.debug(
            f"Registering new bean factory name={name}, type={bean_type}, "
            + f"scope={scope}, factory_method={factory_method}"
        )
        meta = BeanProviderMeta(factory_method, name, bean_type, scope, init_params_need_injection)
        return cls.register_bean_def_with_meta(meta)

    @classmethod
    def register_bean_def_with_meta(cls, meta: BeanProviderMeta) -> ScopedBeanDef:
        """Registering bean with meta.

        Parameters
        ----
        meta: BeanProviderMeta
            Bean's meta data.

        Raises
        ----
        Exception
            If the bean duplicated.
        """
        with cls.__global_lock:
            result = cls.__named_factories.setdefault(meta.bean_name, meta)
            if result != meta:
                raise Exception(
                    "Could not register duplicated bean factory for bean "
                    + f"name={meta.bean_name} type={meta.return_type}"
                )
            bean_type = meta.return_type
            factory_method = meta.func
            params_could_injected = (
                parse_needed_injected_params(factory_method)
                if meta.init_params_need_injection is None
                else meta.init_params_need_injection
            )
            deps = list(map(lambda x: BaseBeanDef(x.bean_name, params_could_injected[x]), params_could_injected.keys()))
            if meta.class_method_flag and meta.class_reference is not None:
                deps.append(BaseBeanDef("", meta.class_reference))
            scoped_bean_def = ScopedBeanDefWithThinDeps(
                meta.bean_name,
                bean_type,
                meta.scope,
                deps,
                None,
                hash(factory_method),
                meta.class_method_flag,
                meta.class_reference,
            )
            cls.__bean_dep_chain.register_bean_def(scoped_bean_def)
            return scoped_bean_def.scoped_bean_def

    @staticmethod
    def snake_case(content: str) -> str:
        """
        Convert a string to snake case style

        @parameters
        content
        """

        upper_case_chars = re.findall(r"([A-Z])", content)
        for c_item in upper_case_chars:
            if content[0] == c_item:
                content = c_item.lower() + content[1:]
            content = content.replace(c_item, f"_{c_item.lower()}")
        return content

    @classmethod
    def __class_bean_decorator(cls, func: Type[Any], scope: Scope, name: str) -> Type[Any]:
        """
        Handler method for @bean with class type
        """

        def class_bean_wrapper() -> Any:
            return func()

        bean_name = name
        if bean_name is None or len(bean_name) == 0:
            bean_name = cls.snake_case(func.__name__)
        init_method = func.__init__
        metadata_attr = METHOD_ANNOTATION_REAL_METHOD_REFERENCE
        if metadata_attr in init_method.__annotations__:
            init_method = init_method.__annotations__[metadata_attr]
        params_need_injected = parse_needed_injected_params(init_method)
        meta = BeanProviderMeta(class_bean_wrapper, bean_name, func, scope, params_need_injected)
        class_bean_wrapper.__annotations__[METHOD_ANNOTATION_META_BEAN] = meta
        class_path = f"{func.__module__}.{func.__name__}"
        cls.__class_ref.setdefault(class_path, func)
        param_str = ", ".join([f"{x}:{y}" for (x, y) in params_need_injected.items()])
        logger.debug(
            "Found init params need to be "
            + f"injected: [{param_str}] for class {func}[{class_path}] method {init_method}."
        )
        return func

    @classmethod
    def bean_method_decorator(
        cls, func: Callable | Type[Any], scope: Scope = Scope.SINGLETON, name: str = ""
    ) -> Callable | Type[Any]:
        """
        The handling method for @bean annotation
        """
        # use another method for preparing metadata for a class bean
        if isinstance(func, type):
            return cls.__class_bean_decorator(cast(Type[Any], func), scope, name)
        real_func = func
        metadata_attr = METHOD_ANNOTATION_REAL_METHOD_REFERENCE
        if is_function(real_func) and metadata_attr in real_func.__annotations__:
            real_func = func.__annotations__[metadata_attr]
            logger.debug(f"find real method {real_func.__name__} by {metadata_attr} attr from {func.__name__}")
        annotations = real_func.__annotations__
        bean_name = real_func.__name__ if len(name) == 0 else name
        return_type = annotations["return"]
        (is_class_member, class_reference) = get_method_delcared_class(real_func)
        logger.debug(
            f"Bean method {real_func.__name__} : {type(real_func)} return: {return_type}, "
            + f"is_class_member:{is_class_member}, class_reference:{class_reference}"
        )
        meta = BeanProviderMeta(
            real_func,
            bean_name,
            return_type,
            scope,
            init_params_need_injection=None,
            class_method_flag=is_class_member,
            class_reference=class_reference,
        )
        real_func.__annotations__[METHOD_ANNOTATION_META_BEAN] = meta
        # return as it is
        return func

    @classmethod
    def injected_method_decorator(cls, func: Callable) -> Callable:
        """
        Handling method for @inject

        Parameters
        ----
        func: Callable
            The method needs injection.

        Returns
        ----
        Callable
            The wrapper method which can be used for invoking
        """
        real_func = func
        bean_provider_meta: Optional[BeanProviderMeta] = None
        # handling the annotation being used with @bean
        if isinstance(real_func, BeanProviderMeta):
            bean_provider_meta = cast(BeanProviderMeta, real_func)
            real_func = cast(BeanProviderMeta, func).func
        logger.debug(f"INIT trying to delcare injected method {real_func.__name__}")
        meta = InjectedInstructionMeta(real_func, bean_provider_meta)
        real_func.__annotations__[METHOD_ANNOTATION_META_INJECTED] = meta

        def wrapper(*args, **kwargs) -> Any:
            if INTERNAL_CALL_FLAG in kwargs:
                del kwargs[INTERNAL_CALL_FLAG]
            (real_args, real_kwargs) = cls.inject_arguments(
                real_func, cast(Tuple[Any], args), cast(Dict[str, Any], kwargs)
            )
            if is_debug():
                kw_args_items = [f"{k}='{v}'" for (k, v) in real_kwargs.items()]
                args_str = (",".join([str(x) for x in real_args]),)
                logger.debug(
                    f"About to call {real_func.__name__} with {len(real_args)} args = {args_str} "
                    + f"and {len(real_kwargs)} kwargs = {kw_args_items}"
                )
            return func(*real_args, **real_kwargs)

        wrapper.__annotations__[METHOD_ANNOTATION_REAL_METHOD_REFERENCE] = func
        return wrapper

    @classmethod
    def inject_method(cls, method: Any) -> Callable:
        """Injecting required arguments for the method.

        Parameters
        ----
        method: Any
            The method requires injection.

        Raises
        ----
        Exception
            If failed to do injection.

        Returns
        ----
        Callable: a callable can be invoked directly
        """
        method_type = type(method)
        if method_type not in [InjectedInstructionMeta, BeanProviderMeta]:
            if is_function(method_type):
                method = InjectedInstructionMeta(method)
            else:
                raise Exception(f"Need a injected method, but got {type(method)}.")
        logger.debug(f"Trying to inject method {method}")
        if method_type == BeanProviderMeta:
            method = InjectedInstructionMeta(method.func)

        def wrapper(*args: Any, **kwargs) -> Any:
            raw_method = method.func
            (real_args, real_kwargs) = cls.inject_arguments(  # pylint: disable=unused-variable
                raw_method, cast(Tuple[Any], args), cast(Dict[str, Any], kwargs), method
            )
            real_kwargs[INTERNAL_CALL_FLAG] = True
            result = method(*args, **real_kwargs)
            logger.debug(f"Got {result} from real method {raw_method.__name__}")
            return result

        return wrapper

    @classmethod
    def inject_arguments(
        cls, method: Callable, args: Tuple[Any], kwargs: Dict[str, Any], meta: Optional[InjectedInstructionMeta] = None
    ) -> Tuple[Tuple[Any], Dict[str, Any]]:
        """
        Inject arguments

        Parameters
        ----
        method: Callable
            The method which contains args and kwargs
        args: Tuple[Any]
            The positional arguments
        kwargs:  Dict[str, Any]
            The named arguments
        meta: Optional[InjectedInstructionMeta]
            The meta data for injection.

        Returns
        Tuple(Tuple[Any], Dict[str, Any])
            First element is the args, and second one is kwargs
        """
        method_type = type(method)
        registry = BeanFactory.registry()
        if not is_function(method_type):
            raise Exception(f"Need a function for injection, but got {method}.")
        real_kwargs: Dict[str, Any] = {}
        for key in kwargs:
            real_kwargs[key] = kwargs[key]
        if meta is None and hasattr(method.__annotations__, METHOD_ANNOTATION_META_INJECTED):
            meta = method.__annotations__[METHOD_ANNOTATION_META_INJECTED]
        if meta is None:
            meta = InjectedInstructionMeta(method)
        for bean_name in meta.injected_params.keys():
            bean_alias = bean_name.bean_name
            param_type = meta.injected_params[bean_name]
            param_value = registry.one_by_name_or_type(bean_alias, param_type)
            real_kwargs[bean_name.param_name] = param_value
            logger.debug(f"Trying to inject method {method.__name__} argument {bean_name}={param_value}")
        return (args, real_kwargs)


class BeanInitializer:
    """Bean initailizer, which should be the entry for DI users."""

    _instance = None
    _bean_metas: Dict[ScopedBeanDef, BeanProviderMeta] = {}

    @classmethod
    def get_instance(cls) -> BeanInitializer:
        """Get the singlton of BeanInitializer"""
        if cls._instance is None:
            return BeanInitializer()
        return cls._instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.debug("*********BeanInitializer is creating**********")
            cls._instance = super().__new__(cls)
            cls._instance.bean_metas = {}
            cls._instance.__init__(*args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.bean_metas: Dict[ScopedBeanDef, BeanProviderMeta] = BeanInitializer._bean_metas

    def register_bean_def(self, scoped_bean_def: ScopedBeanDef, bean_meta: BeanProviderMeta):
        """
        Registering definition for bean and its provider(factory)

        Parameters
        ---
        scoped_bean_def: ScopedBeanDef
            The definition for the bean
        bean_meta: BeanProviderMeta
            The definition of how to create a bean
        """
        logger.debug(f"*** Registering entry scoped_bean_def={scoped_bean_def}, bean_provider_meta={bean_meta}")
        self.bean_metas.setdefault(scoped_bean_def, bean_meta)
        item = bean_meta
        logger.debug(
            f"Save bean provider, real method is {item.func.__name__} scoped bean def "
            + f"type {type(scoped_bean_def)} definition {scoped_bean_def}, bean meta is {bean_meta}"
        )

    def __logging_bean_def(self):
        if not is_debug():
            return
        logger.debug(f"About to print existing bean definitions total={len(self.bean_metas.keys())}")
        for key in list(self.bean_metas.keys()):
            logger.debug(f"[bean def: {key}], [bean provider: {self.bean_metas[key]}]")

    def initialize(self):
        """Initializing the beans"""
        self.__logging_bean_def()
        bean_dep_chain = BeanFactory.bean_dep_chain()
        bean_dep_chain.prepare_for_bean_creation()
        bean_def_list_in_order = bean_dep_chain.get_bean_creation_order()
        registry = BeanFactory.registry()
        for bean_def in bean_def_list_in_order:
            logger.debug(f"Trying to init bean {bean_def}")
            if not bean_def in self.bean_metas:
                raise Exception(f"Could not find BeanProviderMeta for {bean_def}")
            meta = self.bean_metas[bean_def]
            if bean_def.scope == Scope.SINGLETON:
                if callable(meta):
                    result = meta()
                    logger.debug(f"Inited bean {meta.bean_name}[{meta.return_type}]={result}")
                    registry.register_bean(Bean(meta.bean_name, meta.return_type, result, meta))
                else:
                    raise Exception(f"Could not init bean {bean_def} while the meta {meta} is not callable")
            else:
                # save the bean as prototype
                registry.register_bean(PrototypeBean(meta.bean_name, meta.return_type, meta))


def injected(func: Callable) -> Callable:
    """A annotation can be used on any method which wants to inject arguments by magic."""

    return BeanFactory.injected_method_decorator(func)


def bean(scope: Scope = Scope.SINGLETON, name: str = "") -> Callable:
    """
    A annotation can be used for specify factory method for a bean. Borrowed idea from Java.
    Difference name means different bean.

    Parameters
    ----
    scope: Scope
        The bean's scope. SINGLETON or PROTOTYPE. DI will construct single instance of the bean for SINGLETON,
        and multiple instances for PROTOTYPE(create new bean every time you request for it).
    name: str
        The bean's name.
    """

    def decorator(func: Callable | Type[Any]) -> Callable:
        return BeanFactory.bean_method_decorator(func=func, scope=scope, name=name)

    return decorator


def enable_di(func: Callable) -> Callable:
    """Annotation for class main method"""

    def wrapper(*args, **kwargs):
        BeanInitializer.get_instance().initialize()
        return func(*args, **kwargs)

    return wrapper
