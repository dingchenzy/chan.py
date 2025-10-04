"""
缠论分析主模块

本模块实现了缠论分析的核心功能，包括K线数据的加载、处理、分析以及买卖点识别等功能。
缠论是一种技术分析方法，通过对K线形态的分析，识别市场的走势和可能的买卖点。
"""

import copy
import datetime
import pickle
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

from BuySellPoint.BS_Point import CBS_Point  # 买卖点类
from ChanConfig import CChanConfig  # 缠论配置类
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE  # 枚举类型：复权类型、数据源、K线类型
from Common.ChanException import CChanException, ErrCode  # 异常处理类
from Common.CTime import CTime  # 时间处理类
from Common.func_util import check_kltype_order, kltype_lte_day  # 工具函数
from DataAPI.CommonStockAPI import CCommonStockApi  # 通用股票API接口
from KLine.KLine_List import CKLine_List  # K线列表类
from KLine.KLine_Unit import CKLine_Unit  # K线单元类


class CChan:
    """
    缠论分析主类
    
    该类是缠论分析的核心类，负责加载和处理K线数据，并进行缠论分析。
    可以处理多个级别的K线数据，并建立它们之间的关系。
    """
    def __init__(
        self,
        code,                                                # 股票/期货代码
        begin_time=None,                                     # 开始时间
        end_time=None,                                       # 结束时间
        data_src: Union[DATA_SRC, str] = DATA_SRC.BAO_STOCK, # 数据源
        lv_list=None,                                        # K线级别列表
        config=None,                                         # 配置对象
        autype: AUTYPE = AUTYPE.QFQ,                         # 复权类型，默认前复权
    ):
        """
        初始化缠论分析对象
        
        Args:
            code: 股票/期货代码
            begin_time: 开始时间
            end_time: 结束时间
            data_src: 数据源，默认为BAO_STOCK
            lv_list: K线级别列表，默认为[日线, 60分钟线]
            config: 配置对象，默认为None
            autype: 复权类型，默认为前复权
        """
        if lv_list is None:
            lv_list = [KL_TYPE.K_DAY, KL_TYPE.K_60M]
        check_kltype_order(lv_list)  # lv_list顺序从高到低
        self.code = code
        self.begin_time = str(begin_time) if isinstance(begin_time, datetime.date) else begin_time
        self.end_time = str(end_time) if isinstance(end_time, datetime.date) else end_time
        self.autype = autype
        self.data_src = data_src
        self.lv_list: List[KL_TYPE] = lv_list

        if config is None:
            config = CChanConfig()
        self.conf = config

        self.kl_misalign_cnt = 0                    # K线不对齐计数
        self.kl_inconsistent_detail = defaultdict(list)  # K线不一致详情

        self.g_kl_iter = defaultdict(list)          # K线迭代器字典

        self.do_init()                              # 初始化K线数据

        # 如果不是触发步进模式，则一次性加载所有数据
        if not config.trigger_step:
            for _ in self.load():
                ...

    def __deepcopy__(self, memo):
        """
        深拷贝方法
        
        实现对象的深拷贝，确保所有K线数据及其关系都被正确复制
        
        Args:
            memo: 用于记录已复制对象的字典，避免循环引用
            
        Returns:
            CChan: 深拷贝后的对象
        """
        cls = self.__class__
        obj: CChan = cls.__new__(cls)
        memo[id(self)] = obj
        
        # 复制基本属性
        obj.code = self.code
        obj.begin_time = self.begin_time
        obj.end_time = self.end_time
        obj.autype = self.autype
        obj.data_src = self.data_src
        obj.lv_list = copy.deepcopy(self.lv_list, memo)
        obj.conf = copy.deepcopy(self.conf, memo)
        obj.kl_misalign_cnt = self.kl_misalign_cnt
        obj.kl_inconsistent_detail = copy.deepcopy(self.kl_inconsistent_detail, memo)
        obj.g_kl_iter = copy.deepcopy(self.g_kl_iter, memo)
        
        # 复制可能存在的缓存属性
        if hasattr(self, 'klu_cache'):
            obj.klu_cache = copy.deepcopy(self.klu_cache, memo)
        if hasattr(self, 'klu_last_t'):
            obj.klu_last_t = copy.deepcopy(self.klu_last_t, memo)
            
        # 复制K线数据
        obj.kl_datas = {}
        for kl_type, ckline in self.kl_datas.items():
            obj.kl_datas[kl_type] = copy.deepcopy(ckline, memo)
            
        # 重建K线之间的关系（上下级关系）
        for kl_type, ckline in self.kl_datas.items():
            for klc in ckline:
                for klu in klc.lst:
                    assert id(klu) in memo
                    if klu.sup_kl:
                        memo[id(klu)].sup_kl = memo[id(klu.sup_kl)]
                    memo[id(klu)].sub_kl_list = [memo[id(sub_kl)] for sub_kl in klu.sub_kl_list]
        return obj

    def do_init(self):
        """
        初始化K线数据结构
        
        为每个K线级别创建K线列表对象
        """
        self.kl_datas: Dict[KL_TYPE, CKLine_List] = {}
        for idx in range(len(self.lv_list)):
            # idx 代表各个级别，CKLine_List 返回所有集合
            self.kl_datas[self.lv_list[idx]] = CKLine_List(self.lv_list[idx], conf=self.conf)

    def load_stock_data(self, stockapi_instance: CCommonStockApi, lv) -> Iterable[CKLine_Unit]:
        """
        加载股票数据
        
        Args:
            stockapi_instance: 股票API实例
            lv: K线级别
            
        Returns:
            Iterable[CKLine_Unit]: K线单元迭代器
        """
        for KLU_IDX, klu in enumerate(stockapi_instance.get_kl_data()):
            klu.set_idx(KLU_IDX)  # 设置K线索引
            klu.kl_type = lv      # 设置K线级别
            yield klu

    def get_load_stock_iter(self, stockapi_cls, lv):
        """
        获取加载股票数据的迭代器
        
        Args:
            stockapi_cls: 股票API类
            lv: K线级别
            
        Returns:
            迭代器: 用于获取K线数据的迭代器
        """
        stockapi_instance = stockapi_cls(code=self.code, k_type=lv, begin_date=self.begin_time, end_date=self.end_time, autype=self.autype)
        return self.load_stock_data(stockapi_instance, lv)

    def add_lv_iter(self, lv_idx, iter):
        """
        添加级别迭代器
        
        将指定级别的迭代器添加到迭代器字典中
        
        Args:
            lv_idx: 级别索引或级别类型
            iter: 迭代器
        """
        if isinstance(lv_idx, int):
            self.g_kl_iter[self.lv_list[lv_idx]].append(iter)
        else:
            self.g_kl_iter[lv_idx].append(iter)

    def get_next_lv_klu(self, lv_idx):
        """
        获取下一个级别的K线单元
        
        Args:
            lv_idx: 级别索引或级别类型
            
        Returns:
            CKLine_Unit: 下一个K线单元
            
        Raises:
            StopIteration: 当没有更多K线单元时
        """
        if isinstance(lv_idx, int):
            lv_idx = self.lv_list[lv_idx]
        if len(self.g_kl_iter[lv_idx]) == 0:
            raise StopIteration
        try:
            return self.g_kl_iter[lv_idx][0].__next__()
        except StopIteration:
            self.g_kl_iter[lv_idx] = self.g_kl_iter[lv_idx][1:]
            if len(self.g_kl_iter[lv_idx]) != 0:
                return self.get_next_lv_klu(lv_idx)
            else:
                raise

    def step_load(self):
        """
        步进加载数据
        
        在回测或模拟交易场景下，逐步加载数据并返回每一步的状态
        
        Returns:
            迭代器: 每一步的CChan对象状态
        """
        assert self.conf.trigger_step
        self.do_init()  # 清空数据，防止再次重跑没有数据
        yielded = False  # 是否曾经返回过结果
        for idx, snapshot in enumerate(self.load(self.conf.trigger_step)):
            if idx < self.conf.skip_step:
                continue
            yield snapshot
            yielded = True
        if not yielded:
            yield self

    def trigger_load(self, inp):
        """
        触发式加载数据
        
        通过外部传入的K线数据进行加载和计算
        
        Args:
            inp: 输入的K线数据，格式为 {K线级别: [K线单元, ...]}
        """
        # {type: [klu, ...]}
        if not hasattr(self, 'klu_cache'):
            self.klu_cache: List[Optional[CKLine_Unit]] = [None for _ in self.lv_list]
        if not hasattr(self, 'klu_last_t'):
            self.klu_last_t = [CTime(1980, 1, 1, 0, 0) for _ in self.lv_list]
        for lv_idx, lv in enumerate(self.lv_list):
            if lv not in inp:
                if lv_idx == 0:
                    raise CChanException(f"最高级别{lv}没有传入数据", ErrCode.NO_DATA)
                continue
            for klu in inp[lv]:
                klu.kl_type = lv
            assert isinstance(inp[lv], list)
            self.add_lv_iter(lv, iter(inp[lv]))
        for _ in self.load_iterator(lv_idx=0, parent_klu=None, step=False):
            ...
        if not self.conf.trigger_step:  # 非回放模式全部算完之后才算一次中枢和线段
            for lv in self.lv_list:
                self.kl_datas[lv].cal_seg_and_zs()

    def init_lv_klu_iter(self, stockapi_cls):
        """
        初始化各级别K线迭代器
        
        为每个K线级别创建数据迭代器，并处理可能的数据获取失败情况
        
        Args:
            stockapi_cls: 股票API类
            
        Returns:
            list: 有效的K线迭代器列表
        """
        # 为了跳过一些获取数据失败的级别
        lv_klu_iter = []
        valid_lv_list = []
        for lv in self.lv_list:
            try:
                lv_klu_iter.append(self.get_load_stock_iter(stockapi_cls, lv))
                valid_lv_list.append(lv)
            except CChanException as e:
                if e.errcode == ErrCode.SRC_DATA_NOT_FOUND and self.conf.auto_skip_illegal_sub_lv:
                    if self.conf.print_warning:
                        print(f"[WARNING-{self.code}]{lv}级别获取数据失败，跳过")
                    del self.kl_datas[lv]
                    continue
                raise e
        self.lv_list = valid_lv_list
        return lv_klu_iter

    def GetStockAPI(self):
        """
        获取股票API类
        
        根据配置的数据源类型返回对应的API类
        
        Returns:
            类: 股票API类
            
        Raises:
            CChanException: 当数据源类型错误时
        """
        _dict = {}
        if self.data_src == DATA_SRC.BAO_STOCK:
            from DataAPI.BaoStockAPI import CBaoStock
            _dict[DATA_SRC.BAO_STOCK] = CBaoStock
        elif self.data_src == DATA_SRC.CCXT:
            from DataAPI.ccxt import CCXT
            _dict[DATA_SRC.CCXT] = CCXT
        elif self.data_src == DATA_SRC.CSV:
            from DataAPI.csvAPI import CSV_API
            _dict[DATA_SRC.CSV] = CSV_API
        if self.data_src in _dict:
            return _dict[self.data_src]
        assert isinstance(self.data_src, str)
        if self.data_src.find("custom:") < 0:
            raise CChanException("load src type error", ErrCode.SRC_DATA_TYPE_ERR)
        package_info = self.data_src.split(":")[1]
        package_name, cls_name = package_info.split(".")
        import importlib
        module = importlib.import_module(f"DataAPI.{package_name}")
        return getattr(module, cls_name)

    def load(self, step=False):
        """
        加载数据
        
        加载所有级别的K线数据，并进行缠论分析
        
        Args:
            step: 是否为步进模式，默认为False
            
        Returns:
            迭代器: 在步进模式下返回每一步的状态
            
        Raises:
            CChanException: 当最高级别没有数据时
        """
        stockapi_cls = self.GetStockAPI()
        try:
            stockapi_cls.do_init()
            for lv_idx, klu_iter in enumerate(self.init_lv_klu_iter(stockapi_cls)):
                self.add_lv_iter(lv_idx, klu_iter)
            self.klu_cache: List[Optional[CKLine_Unit]] = [None for _ in self.lv_list]
            self.klu_last_t = [CTime(1980, 1, 1, 0, 0) for _ in self.lv_list]

            yield from self.load_iterator(lv_idx=0, parent_klu=None, step=step)  # 计算入口
            if not step:  # 非回放模式全部算完之后才算一次中枢和线段
                for lv in self.lv_list:
                    self.kl_datas[lv].cal_seg_and_zs()
        except Exception:
            raise
        finally:
            stockapi_cls.do_close()
        if len(self[0]) == 0:
            raise CChanException("最高级别没有获得任何数据", ErrCode.NO_DATA)

    def set_klu_parent_relation(self, parent_klu, kline_unit, cur_lv, lv_idx):
        """
        设置K线单元的父子关系
        
        Args:
            parent_klu: 父级K线单元
            kline_unit: 当前K线单元
            cur_lv: 当前K线级别
            lv_idx: 级别索引
        """
        if self.conf.kl_data_check and kltype_lte_day(cur_lv) and kltype_lte_day(self.lv_list[lv_idx-1]):
            self.check_kl_consitent(parent_klu, kline_unit)
        parent_klu.add_children(kline_unit)
        kline_unit.set_parent(parent_klu)

    def add_new_kl(self, cur_lv: KL_TYPE, kline_unit):
        """
        添加新的K线单元
        
        Args:
            cur_lv: 当前K线级别
            kline_unit: K线单元
            
        Raises:
            Exception: 当添加K线单元时发生错误
        """
        try:
            self.kl_datas[cur_lv].add_single_klu(kline_unit)
        except Exception:
            if self.conf.print_err_time:
                print(f"[ERROR-{self.code}]在计算{kline_unit.time}K线时发生错误!")
            raise

    def try_set_klu_idx(self, lv_idx: int, kline_unit: CKLine_Unit):
        """
        尝试设置K线单元的索引
        
        如果K线单元没有索引，则设置一个新的索引
        
        Args:
            lv_idx: 级别索引
            kline_unit: K线单元
        """
        if kline_unit.idx >= 0:
            return
        if len(self[lv_idx]) == 0:
            kline_unit.set_idx(0)
        else:
            kline_unit.set_idx(self[lv_idx][-1][-1].idx + 1)

    def load_iterator(self, lv_idx, parent_klu, step):
        """
        加载迭代器
        
        递归加载各级别的K线数据，并建立它们之间的关系
        
        Args:
            lv_idx: 级别索引
            parent_klu: 父级K线单元
            step: 是否为步进模式
            
        Returns:
            迭代器: 在步进模式下返回每一步的状态
        """
        # K线时间天级别以下描述的是结束时间，如60M线，每天第一根是10点30的
        # 天以上是当天日期
        cur_lv = self.lv_list[lv_idx]
        pre_klu = self[lv_idx][-1][-1] if len(self[lv_idx]) > 0 and len(self[lv_idx][-1]) > 0 else None
        
        
        while True:
            if self.klu_cache[lv_idx]:
                kline_unit = self.klu_cache[lv_idx]
                assert kline_unit is not None
                self.klu_cache[lv_idx] = None
            else:
                try:
                    kline_unit = self.get_next_lv_klu(lv_idx)
                    self.try_set_klu_idx(lv_idx, kline_unit)
                    if not kline_unit.time > self.klu_last_t[lv_idx]:
                        raise CChanException(f"kline time err, cur={kline_unit.time}, last={self.klu_last_t[lv_idx]}, or refer to quick_guide.md, try set auto=False in the CTime returned by your data source class", ErrCode.KL_NOT_MONOTONOUS)
                    self.klu_last_t[lv_idx] = kline_unit.time
                except StopIteration:
                    break

            if parent_klu and kline_unit.time > parent_klu.time:
                self.klu_cache[lv_idx] = kline_unit
                break
            kline_unit.set_pre_klu(pre_klu)
            pre_klu = kline_unit
            self.add_new_kl(cur_lv, kline_unit)
            if parent_klu:
                self.set_klu_parent_relation(parent_klu, kline_unit, cur_lv, lv_idx)
            if lv_idx != len(self.lv_list)-1:
                for _ in self.load_iterator(lv_idx+1, kline_unit, step):
                    ...
                self.check_kl_align(kline_unit, lv_idx)
            if lv_idx == 0 and step:
                yield self

    def check_kl_consitent(self, parent_klu, sub_klu):
        """
        检查K线时间一致性
        
        检查父级别和子级别K线的日期是否一致
        
        Args:
            parent_klu: 父级K线单元
            sub_klu: 子级K线单元
            
        Raises:
            CChanException: 当不一致的K线数量超过配置的阈值时
        """
        if parent_klu.time.year != sub_klu.time.year or \
           parent_klu.time.month != sub_klu.time.month or \
           parent_klu.time.day != sub_klu.time.day:
            self.kl_inconsistent_detail[str(parent_klu.time)].append(sub_klu.time)
            print(f"[WARNING-{self.code}]父级别时间是{parent_klu.time}，次级别时间却是{sub_klu.time}")
            if self.conf.print_warning:
                print(f"[WARNING-{self.code}]父级别时间是{parent_klu.time}，次级别时间却是{sub_klu.time}")
            if len(self.kl_inconsistent_detail) >= self.conf.max_kl_inconsistent_cnt:
                raise CChanException(f"父&子级别K线时间不一致条数超过{self.conf.max_kl_inconsistent_cnt}！！", ErrCode.KL_TIME_INCONSISTENT)

    def check_kl_align(self, kline_unit, lv_idx):
        """
        检查K线对齐
        
        检查当前K线单元是否在次级别找到对应的K线
        
        Args:
            kline_unit: 当前K线单元
            lv_idx: 级别索引
            
        Raises:
            CChanException: 当找不到对应K线的数量超过配置的阈值时
        """
        if self.conf.kl_data_check and len(kline_unit.sub_kl_list) == 0:
            self.kl_misalign_cnt += 1
            if self.conf.print_warning:
                print(f"[WARNING-{self.code}]当前{kline_unit.time}没在次级别{self.lv_list[lv_idx+1]}找到K线！！")
            if self.kl_misalign_cnt >= self.conf.max_kl_misalgin_cnt:
                raise CChanException(f"在次级别找不到K线条数超过{self.conf.max_kl_misalgin_cnt}！！", ErrCode.KL_DATA_NOT_ALIGN)

    def __getitem__(self, n) -> CKLine_List:
        """
        获取指定级别的K线列表
        
        Args:
            n: 级别索引或级别类型
            
        Returns:
            CKLine_List: K线列表
            
        Raises:
            CChanException: 当查询类型不支持时
        """
        if isinstance(n, KL_TYPE):
            return self.kl_datas[n]
        elif isinstance(n, int):
            return self.kl_datas[self.lv_list[n]]
        else:
            raise CChanException("unspoourt query type", ErrCode.COMMON_ERROR)

    def get_bsp(self, idx=None) -> List[CBS_Point]:
        """
        获取买卖点列表（已废弃，请使用get_latest_bsp）
        
        Args:
            idx: 级别索引，如果为None则使用第一个级别
            
        Returns:
            List[CBS_Point]: 买卖点列表
        """
        print('[deprecated] use get_latest_bsp instead')
        if idx is not None:
            return self[idx].bs_point_lst.getSortedBspList()
        assert len(self.lv_list) == 1
        return self[0].bs_point_lst.getSortedBspList()

    def get_latest_bsp(self, idx=None, number=1) -> List[CBS_Point]:
        """
        获取最新的买卖点列表
        
        Args:
            idx: 级别索引，如果为None则使用第一个级别
            number: 返回的买卖点数量，0表示返回全部
            
        Returns:
            List[CBS_Point]: 买卖点列表，从最新到最旧排序
        """
        # number=0则取全部bsp，从最新到最旧排序
        if idx is not None:
            return self[idx].bs_point_lst.get_latest_bsp(number)
        # assert len(self.lv_list) == 1
        return self[0].bs_point_lst.get_latest_bsp(number)

    def chan_dump_pickle(self, file_path):
        """
        将缠论对象保存为pickle文件
        
        在保存前会清除一些循环引用，以便正确序列化
        
        Args:
            file_path: 保存路径
        """
        _pre_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(0x100000)
        for kl_list in self.kl_datas.values():
            for klc in kl_list.lst:
                for klu in klc.lst:
                    klu.pre = None
                    klu.next = None
                klc.set_pre(None)
                klc.set_next(None)
            for bi in kl_list.bi_list:
                bi.pre = None
                bi.next = None
            for seg in kl_list.seg_list:
                seg.pre = None
                seg.next = None

            for segseg in kl_list.segseg_list:
                segseg.pre = None
                segseg.next = None

        with open(file_path, "wb") as f:
            pickle.dump(self, f)

        sys.setrecursionlimit(_pre_limit)

    @staticmethod
    def chan_load_pickle(file_path) -> 'CChan':
        """
        从pickle文件加载缠论对象
        
        加载后会重建对象之间的引用关系
        
        Args:
            file_path: 文件路径
            
        Returns:
            CChan: 加载的缠论对象
        """
        with open(file_path, "rb") as f:
            chan = pickle.load(f)
        last_klu = None
        last_klc = None
        last_bi = None
        last_seg = None
        last_segseg = None
        for kl_list in chan.kl_datas.values():
            for klc in kl_list.lst:
                for klu in klc.lst:
                    klu.pre = last_klu
                    if last_klu:
                        last_klu.next = klu
                    last_klu = klu
                klc.set_pre(last_klc)
                if last_klc:
                    last_klc.set_next(klc)
                last_klc = klc
            for bi in kl_list.bi_list:
                bi.pre = last_bi
                if last_bi:
                    last_bi.next = bi
                last_bi = bi
            for seg in kl_list.seg_list:
                seg.pre = last_seg
                if last_seg:
                    last_seg.next = seg
                last_seg = seg
            for segseg in kl_list.segseg_list:
                segseg.pre = last_segseg
                if last_segseg:
                    last_segseg.next = segseg
                last_segseg = segseg

        return chan
