from __future__ import annotations

import inspect
import logging
import os
import sys
from collections.abc import Callable
from enum import Enum
from threading import RLock
from typing import Any, Generic, List, Optional, TypeVar, cast, Dict, Tuple, Type
from inspect import _empty

InjectedType = TypeVar("InjectedType")
# copied from python dependency injector
if sys.version_info < (3, 7):
    from typing import GenericMeta
else:

    class GenericMeta(type):
        ...


class ClassGetItemMeta(GenericMeta):
    def __getitem__(cls, item):
        # Spike for Python 3.6
        if isinstance(item, tuple):
            return cls(*item)
        return cls(item)


class ClassFqn:
    def __init__(self, module, class_name):
        self.module = module
        self.class_name = class_name
        self.fqn = "{}.{}".format(module, class_name)

    def __str__(self):
        return "ClassFqn[{}]".format(self.fqn)


class Injected(Generic[InjectedType], metaclass=ClassGetItemMeta):
    """
    Injected meta for functions to mark a argument value need to be injected.
    CAUTION: be aware of your argument name, the framework will try to lookup bean by argument name first.

    Parameters
    ----
    type: type, require
        The type of the argument
    """

    def __init__(self, t: Type[Any]):
        self.real_type = t


def is_debug() -> bool:
    """
    Check if the logger level is debug

    Returns
    ---
    bool
        True if the log level is debug, False otherwise
    """
    return logger.root.level == logging.DEBUG


def is_function(item: Any) -> bool:
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
    return isinstance(item, Callable) if type(item) is not type else issubclass(item, Callable)


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
        logger.debug("====Method {}".format(fqn))
        if len(parts) == 1:
            return False, None
        container_path = ".".join(parts[0:-1])
        module_name = method.__module__
        logger.debug("====Method {} container path {}.{}".format(fqn, module_name, container_path))
        return True, ClassFqn(module_name, container_path)
    logger.debug("{} {} type={}is not a method ?".format(method.__name__, method, type(method)))
    return False, None


logger = logging
logger.basicConfig(
    level=int(os.getenv("PY_LOG_LEVEL", "20")),
    format="%(name)s - %(levelname)s - %(pathname)s#%(lineno)d - %(message)s",
)

internal_call_arg = "__di_module_call_identifier__"
method_annotation_meta_injected = "__di_method_annotation_meta_injected__"
method_annotation_meta_bean = "__di_method_annotation_meta_bean__"
method_annotation_real_method_reference = "__di_method_annotation_real_method_ref__"


class Scope(Enum):
    """Scope of the bean, used by @bean() annotation, for example
    @bean(Scope.SINGLETON)

    SINGLETON: mark the bean will have only one instance
    PROTOTYPE: mark the bean will give you a new instance when you ask for it
    """

    SINGLETON = 1
    PROTOTYPE = 2


class BaseBeanDef:
    """A basic bean definition.

    Attributes
    ----
    name: str
        Bean's alias
    type: type
        Bean's type
    """

    def __init__(self, name: str, t: Type[Any] | ClassFqn):
        self.name = name
        self.type = t
        # ref by ClassFqn
        self.shadow_type: Optional[ClassFqn] = t if isinstance(t, ClassFqn) else None

    def __eq__(self, another) -> bool:
        return (
            type(another) == BaseBeanDef
            and another.name == self.name
            and another.type == self.type
            and another.shadow_type == self.shadow_type
        )

    def __hash__(self) -> int:
        return hash(self.__class__) + hash(self.name) + hash(self.type) + hash(self.shadow_type)

    def __str__(self) -> str:
        return "BaseBeanDef(name={}, type={})".format(self.name, self.type)


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
            type(another) == Bean
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

    def __init__(self, name: str, t: Type[Any], meta: BeanProviderMeta):
        super().__init__(name, t, meta, meta)

    def __eq__(self, another) -> bool:
        return (
            type(another) == PrototypeBean
            and another.name == self.name
            and another.type == self.type
            and another.instance == self.instance
        )

    def __hash__(self) -> int:
        return hash(self.__class__) + super().__hash__() + hash(self.meta)

    def __str__(self):
        return "PrototypeBean(name={}, type={}, meta={})".format(self.name, self.type, self.meta)


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
        basic_compare = type(another) == ScopedBeanDef and another.type == self.type and another.scope == self.scope
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
        return "ScopedBeanDef(name={}, type={}, scope={})".format(self.name, self.type, self.scope)


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
    ):
        super().__init__(name, t, scope)
        self.depends_on = depends_on
        self.depended_by = depended_by
        self.scoped_bean_def = ScopedBeanDef(name, t, scope)
        self.method_hash = method_hash
        self.class_method_flag = class_method_flag
        self.class_reference = class_reference

    def __eq__(self, another) -> bool:
        basic_eq = (
            type(another) == ScopedBeanDefWithThinDeps
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
        return "name={}, type={}, scope={}, depended by {}, depends on {}".format(
            self.name, self.type, self.scope, self.depended_by, depends_on_str
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

    def __init__(self):
        self.raw_defs: List[ScopedBeanDefWithThinDeps] = []
        self.raw_defs_map: Dict[BaseBeanDef, ScopedBeanDefWithThinDeps] = dict()
        self.bare_type_and_raw_defs_map: Dict[BaseBeanDef, ScopedBeanDef] = dict()
        self.init_order_map: dict[BaseBeanDef, int] = dict()

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
            for k in keys:
                logger.debug("Will init {} at order {}.".format(k, self.init_order_map[k]))
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
            "Registering bean definition {} hash={} scoped-bean-def-hash={}".format(
                bean_def, hash(bean_def), hash(scoped_bean_def)
            )
        )
        self.raw_defs.append(bean_def)
        current_value = self.raw_defs_map.setdefault(scoped_bean_def, bean_def)
        # use protype scope if there's conflicts happened
        if current_value != bean_def and bean_def.scope == Scope.PROTOTYPE:
            logger.debug("Updating value in raw_defs_map {} to {}".format(scoped_bean_def, bean_def))
            self.raw_defs_map[scoped_bean_def] = bean_def

        bare_type_raw_def = self.bare_type_and_raw_defs_map.setdefault(bean_def.base_bean_def, scoped_bean_def)
        # use protype scope if there's conflicts happened
        if bare_type_raw_def != bean_def and bean_def.scope == Scope.PROTOTYPE:
            logger.debug("Updating value in bare_type_and_raw_defs_map {} to {}".format(bean_def.type, bean_def))
            self.bare_type_and_raw_defs_map[bean_def.base_bean_def] = scoped_bean_def
        return self

    def increase_type_init_order(
        self, t: BaseBeanDef, qty: int, update_depdencies: bool = True, path: List[BaseBeanDef] = []
    ):
        """Increase bean's init order, can be used as decrease init order by
        passing negative order.

        Parameters
        ----
        t: BaseBeanDef, required
            The bean which will be affected
        order: int, required
            The order value affected. Positive value for marking the bean will be inited latter.
        update_depdencies: bool, optional, defualt is True
            Whether to update bean's dependencies as well.
        path: List[BaseBeanDef], optional
            The dependencies path of current BaseBeanDef.

        Raises
        ----
        Exception
            If circular dependency found, or the BaseBeanDef could not be found,
            or bean's dependencies could not be found.
        """
        old_order = self.init_order_map[t] if t in self.init_order_map else 0
        final_order = int(qty) + int(old_order)
        self.init_order_map[t] = final_order
        logger.debug(
            "Bean {} init order is {},  dep path: {}".format(t, final_order, " > ".join([str(x) for x in path]))
        )
        if not update_depdencies:
            return
        self.prepare_dependencies(t, abs(qty), path)

    def logging_raw_def_maps(self):
        if not is_debug():
            return
        logger.debug("Will print registered bean")
        keys = list(self.raw_defs_map.keys())
        for t in keys:
            definition = self.raw_defs_map[t]
            logger.debug("type {} = {} hash={}".format(t, definition, hash(t)))

    def logging_bean_init_order(self):
        if not is_debug():
            return
        definitions = self.get_bean_creation_order()
        order = 0
        for item in definitions:
            logger.debug("[{}] the init order for {} is {}".format(order, item, order))
            order = order + 1

    def prepare_dependencies(
        self, base_bean_def: BaseBeanDef, base_amount: int = 1, existing_call_path: List[BaseBeanDef] = []
    ):
        """
        Prepare the dependencies for base_bean_def, so its dependencies can be inited in proper order.

        base_bean_def: BaseBeanDef
            The start point for dependencies
        base_amount: int
            The init order affected
        existing_call_path: List[BaseBeanDef]
            The dependencies path, for detecting circular dependencies
        """
        t = base_bean_def
        # todo: may find a better way to do it?
        if existing_call_path.count(t) > 1:
            path_temp = [x for x in existing_call_path]
            path_temp.append(t)
            dep_list = ["{}".format(x) for x in path_temp]
            raise Exception("Circular dependencies found: {}".format(" -> ".join(dep_list)))
        if t not in self.raw_defs_map:
            raise Exception("Could not find {} hash={} from bean definition map".format(t, hash(t)))
        bean_def = self.raw_defs_map[t]
        # increase init order for bean which needs this type
        if bean_def.depended_by is not None:
            tmp_path = [t]
            tmp_path.extend(existing_call_path)
            self.increase_type_init_order(bean_def, base_amount * -1, False, tmp_path)
            self.increase_type_init_order(bean_def.depended_by, base_amount, True, tmp_path)
        depends_on = bean_def.depends_on
        if depends_on is None or len(depends_on) == 0:
            return
        bare_types = self.bare_type_and_raw_defs_map.keys()
        for dep_bean_def in depends_on:
            if type(dep_bean_def.type) == ClassFqn:
                old_type = dep_bean_def
                fetched_type = BeanFactory.get_class(cast(ClassFqn, dep_bean_def.type))
                ref_type = dep_bean_def.type
                dep_bean_def = BaseBeanDef(BeanFactory.snake_case(fetched_type.__name__), fetched_type)
                dep_bean_def.shadow_type = cast(ClassFqn, ref_type)
                logger.debug("Corrected depends on from {} to {}".format(old_type, dep_bean_def))
            if dep_bean_def not in self.bare_type_and_raw_defs_map:
                # trying to find any subclass can work
                filter_method = lambda x: issubclass(x.type, cast(Type, dep_bean_def.type))
                found_types = list(filter(filter_method, bare_types))
                if len(found_types) == 0:
                    raise Exception(
                        "Failed to find bean definition of {} which is depended by {}".format(bean_def.type, t.type)
                    )
                # use one that will work
                # trying to find by bean name, or use matched first item
                # todo: may also check bean's type as well
                result_by_name = list(filter(lambda x: x.name == bean_def.name, found_types))
                old_type = dep_bean_def
                dep_bean_def = found_types[0] if len(result_by_name) == 0 else result_by_name[0]
                logger.debug(
                    "Chosen matched type {} for {} from {}".format(
                        dep_bean_def, old_type, ", ".join([str(x) for x in found_types])
                    )
                )
            the_bean_def = self.bare_type_and_raw_defs_map[dep_bean_def]
            tmp_path = [x for x in existing_call_path]
            tmp_path.append(t)
            amount = -1 * base_amount
            logger.debug(
                "Trying to increate order for {} by {} because it is needed by {}, calling path = {}".format(
                    the_bean_def, amount, t, tmp_path
                )
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
        [self.increase_type_init_order(x, 0, False) for x in all_types]
        for t in all_types:
            self.prepare_dependencies(t)
        self.logging_bean_init_order()
        return self


class BeanRegistry:
    """Bean's registry, can be used for looking up a bean without injection
    point declared."""

    _instance: Optional[BeanRegistry] = None
    _beans_name_map: dict[str, Bean] = dict()
    _beans_type_map: dict[BaseBeanDef, List[Bean]] = dict()
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

    def __init__(self):
        self.__beans_name_map: dict[str, Bean] = BeanRegistry._beans_name_map
        self.__beans_type_map: dict[BaseBeanDef, List[Bean]] = BeanRegistry._beans_type_map
        self.__bean_lock = BeanRegistry._bean_lock

    def register_bean(self, bean: Bean):
        """Registering a bean into registry.

        Parameters
        ----
        bean: Bean
            The bean's instance.
        """
        with self.__bean_lock:
            result = self.__beans_name_map.setdefault(bean.name, bean)
            if result != bean:
                raise Exception("Bean of name={} type={} was already existed.".format(bean.name, bean.type))
            bean_list = self.__beans_type_map.setdefault(bean.base_bean_def, list())
            if len(bean_list) == 0 or bean_list.count(bean) == 0:
                bean_list.append(bean)

    def __get_or_create(self, bean: Bean) -> Any:
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
        is_correct_type = isinstance(bean, Bean)
        if bean is None or not is_correct_type:
            raise Exception("Could not find a bean named {}".format(bean.name))
        if type(bean) == PrototypeBean:
            return bean.instance()
        return bean.instance

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
            raise Exception("Could not find a bean named {}".format(name))
        bean = self.__beans_name_map[name]
        return self.__get_or_create(bean)

    def one_by_type(self, t: Type[Any]) -> Any:
        """Get bean by it type.

        Parameters
        ----
        t: type
            The definition for the bean

        Raises
        ----
        Exception
            If the bean was not found or more than one instance found.

        Returns
        ----
        Any: The bean's instance, could be any type.
        """
        items = self.list_by_type(BaseBeanDef("", t))
        if len(items) > 1:
            raise Exception(
                "Could not determine which instance to use for {} from {} candidates.".format(t, len(items))
            )
        return items[0]

    def one_by_name_or_type(self, name: str, t: Type[Any]) -> Any:
        """Get a bean by its name or type.

        Parameters
        ----
        name: str
            Bean's name
        t: type
            Bean's type

        Raises
        ----
        Exception
            If the bean could not be found or more than one instance found.
        """
        try:
            return self.one_by_name(name)
        except:
            return self.one_by_type(t)

    def list_by_type(self, t: BaseBeanDef) -> List[Any]:
        """Find a instances list of specificied bean type.

        Parameters
        ----
        t: BaseBeanDef
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
        if not t in self.__beans_name_map:
            items = self.__get_matching_types(BeanFactory.get_class_by_bean_def(t))
        else:
            items = self.__beans_type_map[t]
            if len(items) == 0:
                items = self.__get_matching_types(t.type)
        if len(items) == 0:
            raise Exception("Could not find a bean for type {}".format(t))
        # trying to matching bean by name first
        name_filter_method = lambda x: x.name == t.name and issubclass(x.type, BeanFactory.get_class_by_bean_def(t))
        result_by_name = list(filter(name_filter_method, items))
        items = result_by_name if len(result_by_name) > 0 else items
        items = [self.__get_or_create(x) for x in items]
        return items

    def __get_matching_types(self, t: Type[Any]) -> List[Bean]:
        """Get a proper list of beans which matching requested type.

        Parameters
        ----
        t: type
            The type or super type for the result.

        Returns
        ----
        List[Bean]: a list of Bean which matching t, could be exact type or sub type.
        """
        if type(t) is BeanProviderMeta:
            t = cast(BeanProviderMeta, t).return_type
        logger.debug("Trying to find matching type for {}, type is {}".format(t, type(t)))
        filter_method = lambda x: issubclass(x.type, t)
        keys = self.__beans_type_map.keys()
        matching_types = list(filter(filter_method, keys))
        getter_method = lambda x: self.__beans_type_map[x]
        result: List[Bean] = []
        [result.extend(getter_method(x)) for x in matching_types]
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


def parse_needed_injected_params(func: Callable) -> Dict[str, Type[Any]]:
    """Parsing params should be injected by DI.

    Parameters
    ----
    func: Callable
        The method

    Returns
    ----
    dict[str, type]: A dict contains name and its type for injecting.
    """
    injected_params: Dict[str, Type[Any]] = dict()
    signature = inspect.signature(func)
    parameters = signature.parameters
    method_name = func.__name__
    logger.debug("function {} has parameters : {} ".format(method_name, parameters))
    for k, v in parameters.items():
        default_value = v.default
        param_type: Optional[Type] = None
        if v.annotation is not None:
            param_type = v.annotation
        if isinstance(default_value, Injected):
            injected_value: Injected = cast(Injected, default_value)
            param_type = injected_value.real_type
        if param_type is None:
            logger.warn(
                "Failed to decide injected data type for {}.{}, annotated type is {} and default value is {}".format(
                    method_name, k, v.annotation, default_value
                )
            )
            continue
        if will_argument_be_ignored(func, k) or param_type is _empty:
            logger.debug("Ignoring argument {} of method {}".format(k, method_name))
            continue
        # logger.debug("*** {}.{} can be injected with type={}".format(method_name, k, param_type))
        injected_params.setdefault(k, param_type)
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
        init_params_need_injection: Optional[Dict[str, Type[Any]]] = None,
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
            "Invoking {} is_class_member ? {}, class_reference = {}, args type={}, kwargs type = {}".format(
                self.func.__name__, self.class_method_flag, self.class_reference, type(args), type(kwargs)
            )
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
                (
                    "Trying to setup self reference for calling method {}, {} args {} kwargs."
                    + "class_type = {}, instance = {}"
                ).format(self.func.__name__, len(args), len(kwargs.keys()), class_type, class_instance)
            )

        return BeanFactory.inject_method(self.func)(*args, **kwargs)

    def do_register(self) -> ScopedBeanDef:
        logger.debug("trying to register with {}".format(self.func.__name__))
        register_result = BeanFactory.register_bean_def_with_meta(self)
        BeanInitializer.get_instance().register_bean_def(register_result, self)
        return register_result

    def __str__(self):
        params_str = (
            "None"
            if self.init_params_need_injection is None
            else ",".join(["{}:{}".format(k, v) for (k, v) in self.init_params_need_injection.items()])
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
        logger.debug(
            "InjectedInstructionMeta {}.__call__ {} {} positional arguments, {} kwargs {}".format(
                self, method.__name__, len(args), len(kwargs), ",".join([k for k in kwargs])
            )
        )
        # call real method or there will be stack overflow happened.
        if internal_call_arg in kwargs:
            del kwargs[internal_call_arg]
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

    __named_factories: Dict[str, BeanProviderMeta] = dict()
    __class_ref: Dict[str, Type[Any]] = dict()
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
        m = sys.modules[class_path.module]
        return getattr(m, class_path.class_name)

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
        if type(bean_def.type) == ClassFqn:
            return cls.get_class(cast(ClassFqn, bean_def.type))
        return cast(Type[Any], bean_def.type)

    @classmethod
    def register_bean_def(
        cls,
        name: str,
        t: Type[Any],
        factory_method: Callable,
        scope: Scope = Scope.SINGLETON,
        init_params_need_injection: Optional[Dict[str, Type[Any]]] = None,
    ) -> ScopedBeanDef:
        """Registering bean's definition.

        Parameters
        ----
        name: str
            Bean's type
        t: type
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
            "Registering new bean factory name={}, type={}, scope={}, factory_method={}".format(
                name, t, factory_method, scope
            )
        )
        meta = BeanProviderMeta(factory_method, name, t, scope, init_params_need_injection)
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
                    "Could not register duplicated bean factory for bean name={} type={}".format(
                        meta.bean_name, meta.return_type
                    )
                )
            bean_type = meta.return_type
            factory_method = meta.func
            params_could_injected = (
                parse_needed_injected_params(factory_method)
                if meta.init_params_need_injection is None
                else meta.init_params_need_injection
            )
            deps = list(map(lambda x: BaseBeanDef(x, params_could_injected[x]), params_could_injected.keys()))
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
        import re

        upper_case_chars = re.findall(r"([A-Z])", content)
        for c in upper_case_chars:
            if content[0] == c:
                content = c.lower() + content[1:]
            content = content.replace(c, "_{}".format(c.lower()))
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
        metadata_attr = method_annotation_real_method_reference
        if metadata_attr in init_method.__annotations__:
            init_method = init_method.__annotations__[metadata_attr]
        params_need_injected = parse_needed_injected_params(init_method)
        meta = BeanProviderMeta(class_bean_wrapper, bean_name, func, scope, params_need_injected)
        class_bean_wrapper.__annotations__[method_annotation_meta_bean] = meta
        class_path = "{}.{}".format(func.__module__, func.__name__)
        cls.__class_ref.setdefault(class_path, func)
        logger.debug(
            "Found init params need to be injected: [{}] for class {}[{}] method {}.".format(
                ", ".join(["{}:{}".format(x, y) for (x, y) in params_need_injected.items()]),
                func,
                class_path,
                init_method,
            )
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
        if type(func) is type:
            return cls.__class_bean_decorator(cast(Type[Any], func), scope, name)
        real_func = func
        metadata_attr = method_annotation_real_method_reference
        if is_function(real_func) and metadata_attr in real_func.__annotations__:
            real_func = func.__annotations__[metadata_attr]
            logger.debug(
                "find real method {} by {} attr from {}".format(real_func.__name__, metadata_attr, func.__name__)
            )
        annotations = real_func.__annotations__
        bean_name = real_func.__name__ if len(name) == 0 else name
        return_type = annotations["return"]
        (is_class_member, class_reference) = get_method_delcared_class(real_func)
        logger.debug(
            "Bean method {} : {} return: {}, is_class_member:{}, class_reference:{}".format(
                real_func.__name__, type(real_func), return_type, is_class_member, class_reference
            )
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
        real_func.__annotations__[method_annotation_meta_bean] = meta
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
            real_func = func.func
        logger.debug("INIT trying to delcare injected method {}".format(real_func.__name__))
        meta = InjectedInstructionMeta(real_func, bean_provider_meta)
        real_func.__annotations__[method_annotation_meta_injected] = meta

        def wrapper(*args, **kwargs) -> Any:
            if internal_call_arg in kwargs:
                del kwargs[internal_call_arg]
            (real_args, real_kwargs) = cls.inject_arguments(real_func, args, kwargs)
            logger.debug(
                "About to call {} with {} args = {} and {} kwargs = {}".format(
                    real_func.__name__,
                    len(real_args),
                    ",".join([str(x) for x in real_args]),
                    len(real_kwargs.keys()),
                    ["{}='{}'".format(k, v) for (k, v) in real_kwargs.items()],
                )
            )
            return func(*real_args, **real_kwargs)

        wrapper.__annotations__[method_annotation_real_method_reference] = func
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
        if method_type != InjectedInstructionMeta and method_type != BeanProviderMeta:
            if is_function(method_type):
                method = InjectedInstructionMeta(method)
            else:
                raise Exception("Need a injected method, but got {}.".format(type(method)))
        logger.debug("Trying to inject method {}".format(method))
        if method_type == BeanProviderMeta:
            method = InjectedInstructionMeta(method.func)

        def wrapper(*args: Any, **kwargs) -> Any:
            raw_method = method.func
            (_, real_kwargs) = cls.inject_arguments(raw_method, args, kwargs, method)
            real_kwargs[internal_call_arg] = True
            result = method(*args, **real_kwargs)
            logger.debug("Got {} from real method {}".format(result, raw_method.__name__))
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
            raise Exception("Need a function for injection, but got {}.".format(type(method)))
        real_kwargs: Dict[str, Any] = dict()
        for k in kwargs:
            real_kwargs[k] = kwargs[k]
        if meta is None and hasattr(method.__annotations__, method_annotation_meta_injected):
            meta = method.__annotations__[method_annotation_meta_injected]
        if meta is None:
            meta = InjectedInstructionMeta(method)
        for bean_name in meta.injected_params.keys():
            param_type = meta.injected_params[bean_name]
            param_value = registry.one_by_name_or_type(bean_name, param_type)
            real_kwargs[bean_name] = param_value
            logger.debug("Trying to inject method {} argument {}={}".format(method.__name__, bean_name, param_value))
        return (args, real_kwargs)


class BeanInitializer:
    """Bean initailizer, which should be the entry for DI users."""

    _instance = None
    _bean_metas: Dict[ScopedBeanDef, BeanProviderMeta] = dict()

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
            cls._instance.bean_metas = dict()
            cls._instance.__init__(*args, **kwargs)
        return cls._instance

    def __init__(self):
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
        logger.debug(
            "*** Registering entry scoped_bean_def={}, bean_provider_meta={}".format(scoped_bean_def, bean_meta)
        )
        self.bean_metas.setdefault(scoped_bean_def, bean_meta)
        item = bean_meta
        logger.debug(
            "Save bean provider, real method is {} scoped bean def type {} definition {}, bean meta is {}".format(
                item.func.__name__, type(scoped_bean_def), scoped_bean_def, bean_meta
            )
        )

    def __logging_bean_def(self):
        if not is_debug():
            return
        logger.debug("About to print existing bean definitions total={}".format(len(self.bean_metas.keys())))
        for key in list(self.bean_metas.keys()):
            logger.debug("[bean def: {}], [bean provider: {}]".format(key, self.bean_metas[key]))

    def initialize(self):
        """Initializing the beans"""
        self.__logging_bean_def()
        bean_dep_chain = BeanFactory.bean_dep_chain()
        bean_dep_chain.prepare_for_bean_creation()
        bean_def_list_in_order = bean_dep_chain.get_bean_creation_order()
        registry = BeanFactory.registry()
        for bean_def in bean_def_list_in_order:
            logger.debug("Trying to init bean {}".format(bean_def))
            if not bean_def in self.bean_metas:
                raise Exception("Could not find BeanProviderMeta for {}".format(bean_def))
            meta = self.bean_metas[bean_def]
            if bean_def.scope == Scope.SINGLETON:
                if callable(meta):
                    result = meta()
                    logger.debug("Inited bean {}[{}]={}".format(meta.bean_name, meta.return_type, result))
                    registry.register_bean(Bean(meta.bean_name, meta.return_type, result, meta))
                else:
                    raise Exception("Could not init bean {} while the meta {} is not callable".format(bean_def, meta))
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
