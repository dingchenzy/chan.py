"""
线段模块 - 缠论中线段的实现

线段是缠论中的重要概念，由多个笔组成，是比笔更高级别的结构。
线段的形成需要满足特定的条件，包括方向一致性、长度要求等。
"""
from typing import Generic, List, Optional, Self, TypeVar
 
from Bi.Bi import CBi
from Common.CEnum import BI_DIR, MACD_ALGO, TREND_LINE_SIDE
from Common.ChanException import CChanException, ErrCode
from KLine.KLine_Unit import CKLine_Unit
from Math.TrendLine import CTrendLine

from .EigenFX import CEigenFX

# 泛型类型定义，LINE_TYPE可以是CBi（笔）或CSeg（线段）
LINE_TYPE = TypeVar('LINE_TYPE', CBi, "CSeg")


class CSeg(Generic[LINE_TYPE]):
    """
    线段类 - 缠论中线段的核心实现
    
    线段是由多个同方向的笔组成的更高级别结构，是缠论分析的重要组成部分。
    线段具有明确的开始笔和结束笔，以及确定的方向（上升或下降）。
    
    主要功能：
    1. 线段的创建和验证
    2. 线段内中枢的管理
    3. 线段的各种计算（振幅、斜率、MACD指标等）
    4. 线段间的关系维护（前后关系、父子关系）
    5. 买卖点的标记
    """
    
    def __init__(self, idx: int, start_bi: LINE_TYPE, end_bi: LINE_TYPE, is_sure=True, seg_dir=None, reason="normal"):
        """
        初始化线段对象
        
        Args:
            idx: 线段索引，用于标识线段在序列中的位置
            start_bi: 开始笔，线段的起始笔
            end_bi: 结束笔，线段的结束笔
            is_sure: 线段是否确定，False表示线段可能还会延续
            seg_dir: 线段方向，如果为None则使用end_bi的方向
            reason: 线段形成的原因，用于调试和分析
        
        Raises:
            AssertionError: 当笔的方向不一致且线段确定时抛出
        """
        # 验证笔的方向一致性：除了第一个笔或线段不确定的情况，开始笔和结束笔的方向必须一致
        assert start_bi.idx == 0 or start_bi.dir == end_bi.dir or not is_sure, f"{start_bi.idx} {end_bi.idx} {start_bi.dir} {end_bi.dir}"
        
        # 基本属性
        self.idx = idx                    # 线段索引
        self.start_bi = start_bi         # 开始笔
        self.end_bi = end_bi            # 结束笔
        self.is_sure = is_sure          # 线段是否已经结束确定
        self.dir = end_bi.dir if seg_dir is None else seg_dir   # 线段方向

        # 中枢相关
        from ZS.ZS import CZS
        self.zs_lst: List[CZS[LINE_TYPE]] = []  # 属于这个线段的所有中枢列表

        # 特征序列和线段关系
        self.eigen_fx: Optional[CEigenFX] = None    # 特征序列分型，用于更高级别的分析
        self.seg_idx = None                         # 线段的线段索引（当线段作为更高级别线段的组成部分时）
        self.parent_seg: Optional[CSeg] = None      # 父线段，表示当前线段属于哪个更高级别的线段
        self.pre: Optional[Self] = None             # 前一个线段
        self.next: Optional[Self] = None            # 后一个线段

        # 买卖点
        from BuySellPoint.BS_Point import CBS_Point
        self.bsp: Optional[CBS_Point] = None        # 线段尾部的买卖点标记

        # 线段内部结构
        self.bi_list: List[LINE_TYPE] = []          # 线段包含的所有笔，仅通过update_bi_list方法更新
        self.reason = reason                       
        
        # 趋势线
        self.support_trend_line = None              # 支撑趋势线
        self.resistance_trend_line = None           # 阻力趋势线
        
        # 长度验证：线段至少需要包含3个笔（索引差至少为2）
        if end_bi.idx - start_bi.idx < 2:
            self.is_sure = False
            
        # 执行线段有效性检查
        self.check()

        # 内部元素确定性标记
        self.ele_inside_is_sure = False

    def set_seg_idx(self, idx):
        """
        设置线段的线段索引
        
        当线段作为更高级别线段的组成部分时，需要设置其在高级别线段中的索引
        
        Args:
            idx: 线段索引
        """
        self.seg_idx = idx

    def check(self):
        """
        检查线段的有效性
        
        验证线段是否满足缠论的基本要求：
        1. 方向一致性：上升线段的起点应低于终点，下降线段相反
        2. 长度要求：线段至少包含3个笔
        
        Raises:
            CChanException: 当线段不满足有效性要求时抛出异常
        """
        if not self.is_sure:
            return
            
        # 检查方向和价格的一致性
        if self.is_down():
            # 下降线段：起始点应该高于结束点
            if self.start_bi.get_begin_val() < self.end_bi.get_end_val():
                raise CChanException(f"下降线段起始点应该高于结束点! idx={self.idx}", ErrCode.SEG_END_VALUE_ERR)
        elif self.start_bi.get_begin_val() > self.end_bi.get_end_val():
            # 上升线段：起始点应该低于结束点
            raise CChanException(f"上升线段起始点应该低于结束点! idx={self.idx}", ErrCode.SEG_END_VALUE_ERR)
            
        # 检查线段长度
        if self.end_bi.idx - self.start_bi.idx < 2:
            raise CChanException(f"线段({self.start_bi.idx}-{self.end_bi.idx})长度不能小于2! idx={self.idx}", ErrCode.SEG_LEN_ERR)

    def __str__(self):
        """
        线段的字符串表示
        
        Returns:
            str: 包含起始笔索引、结束笔索引、方向和确定性的字符串
        """
        return f"{self.start_bi.idx}->{self.end_bi.idx}: {self.dir}  {self.is_sure}"

    def add_zs(self, zs):
        """
        添加中枢到线段
        
        中枢是反序加入的，最新的中枢会被添加到列表的开头
        
        Args:
            zs: 要添加的中枢对象
        """
        self.zs_lst = [zs] + self.zs_lst  # 因为中枢是反序加入的

    def cal_klu_slope(self):
        """
        计算线段的K线斜率
        
        斜率 = (结束价格 - 开始价格) / (K线数量 * 开始价格)
        用于衡量线段的价格变化速度
        
        Returns:
            float: 线段的斜率值
        """
        assert self.end_bi.idx >= self.start_bi.idx
        return (self.get_end_val()-self.get_begin_val())/(self.get_end_klu().idx-self.get_begin_klu().idx)/self.get_begin_val()

    def cal_amp(self):
        """
        计算线段的振幅（相对变化）
        
        振幅 = (结束价格 - 开始价格) / 开始价格
        
        Returns:
            float: 线段的相对振幅
        """
        return (self.get_end_val()-self.get_begin_val())/self.get_begin_val()

    def cal_bi_cnt(self):
        """
        计算线段包含的笔数量
        
        Returns:
            int: 笔的数量
        """
        return self.end_bi.idx-self.start_bi.idx+1

    def clear_zs_lst(self):
        """
        清空线段的中枢列表
        
        用于重新计算或重置线段的中枢信息
        """
        self.zs_lst = []

    def _low(self):
        """
        获取线段的最低价
        
        Returns:
            float: 线段的最低价格
        """
        return self.end_bi.get_end_klu().low if self.is_down() else self.start_bi.get_begin_klu().low

    def _high(self):
        """
        获取线段的最高价
        
        Returns:
            float: 线段的最高价格
        """
        return self.end_bi.get_end_klu().high if self.is_up() else self.start_bi.get_begin_klu().high

    def is_down(self):
        """
        判断线段是否为下降方向
        
        Returns:
            bool: True表示下降线段，False表示上升线段
        """
        return self.dir == BI_DIR.DOWN

    def is_up(self):
        """
        判断线段是否为上升方向
        
        Returns:
            bool: True表示上升线段，False表示下降线段
        """
        return self.dir == BI_DIR.UP

    def get_end_val(self):
        """
        获取线段结束价格
        
        Returns:
            float: 线段结束笔的结束价格
        """
        return self.end_bi.get_end_val()

    def get_begin_val(self):
        """
        获取线段开始价格
        
        Returns:
            float: 线段开始笔的开始价格
        """
        return self.start_bi.get_begin_val()

    def amp(self):
        """
        获取线段的绝对振幅
        
        Returns:
            float: 线段开始和结束价格的绝对差值
        """
        return abs(self.get_end_val() - self.get_begin_val())

    def get_end_klu(self) -> CKLine_Unit:
        """
        获取线段结束的K线单元
        
        Returns:
            CKLine_Unit: 线段结束笔对应的K线单元
        """
        return self.end_bi.get_end_klu()

    def get_begin_klu(self) -> CKLine_Unit:
        """
        获取线段开始的K线单元
        
        Returns:
            CKLine_Unit: 线段开始笔对应的K线单元
        """
        return self.start_bi.get_begin_klu()

    def get_klu_cnt(self):
        """
        获取线段包含的K线数量
        
        Returns:
            int: 从开始K线到结束K线的总数量
        """
        return self.get_end_klu().idx - self.get_begin_klu().idx + 1

    def cal_macd_metric(self, macd_algo, is_reverse):
        """
        计算MACD相关指标
        
        根据指定的算法计算MACD指标，用于线段的技术分析
        
        Args:
            macd_algo: MACD算法类型（SLOPE斜率或AMP振幅）
            is_reverse: 是否反向计算（暂未使用）
            
        Returns:
            float: 计算得到的MACD指标值
            
        Raises:
            CChanException: 当算法类型不支持时抛出异常
        """
        if macd_algo == MACD_ALGO.SLOPE:
            return self.Cal_MACD_slope()
        elif macd_algo == MACD_ALGO.AMP:
            return self.Cal_MACD_amp()
        else:
            raise CChanException(f"unsupport macd_algo={macd_algo} of Seg, should be one of slope/amp", ErrCode.PARA_ERROR)

    def Cal_MACD_slope(self):
        """
        计算MACD斜率指标
        
        根据线段方向计算相应的斜率：
        - 上升线段：(最高价 - 最低价) / 最高价 / K线数量
        - 下降线段：(最高价 - 最低价) / 最高价 / K线数量
        
        Returns:
            float: MACD斜率值
        """
        begin_klu = self.get_begin_klu()
        end_klu = self.get_end_klu()
        if self.is_up():
            return (end_klu.high - begin_klu.low)/end_klu.high/(end_klu.idx - begin_klu.idx + 1)
        else:
            return (begin_klu.high - end_klu.low)/begin_klu.high/(end_klu.idx - begin_klu.idx + 1)

    def Cal_MACD_amp(self):
        """
        计算MACD振幅指标
        
        根据线段方向计算相应的振幅：
        - 上升线段：(最高价 - 最低价) / 最低价
        - 下降线段：(最高价 - 最低价) / 最高价
        
        Returns:
            float: MACD振幅值
        """
        begin_klu = self.get_begin_klu()
        end_klu = self.get_end_klu()
        if self.is_down():
            return (begin_klu.high-end_klu.low)/begin_klu.high
        else:
            return (end_klu.high-begin_klu.low)/begin_klu.low

    def update_bi_list(self, bi_lst, idx1, idx2):
        """
        更新线段包含的笔列表
        
        将指定范围内的笔添加到线段中，并建立父子关系。
        当笔数量达到3个或以上时，会计算支撑和阻力趋势线。
        
        Args:
            bi_lst: 笔列表
            idx1: 起始笔索引
            idx2: 结束笔索引
        """
        for bi_idx in range(idx1, idx2+1):
            bi_lst[bi_idx].parent_seg = self  # 设置笔的父线段
            self.bi_list.append(bi_lst[bi_idx])
            
        # 当笔数量足够时，计算趋势线
        if len(self.bi_list) >= 3:
            self.support_trend_line = CTrendLine(self.bi_list, TREND_LINE_SIDE.INSIDE)    # 内侧趋势线（支撑线）
            self.resistance_trend_line = CTrendLine(self.bi_list, TREND_LINE_SIDE.OUTSIDE) # 外侧趋势线（阻力线）

    def get_first_multi_bi_zs(self):
        """
        获取第一个多笔中枢
        
        多笔中枢是由多个笔形成的中枢，相对于单笔中枢更加稳定和重要
        
        Returns:
            CZS或None: 第一个多笔中枢，如果不存在则返回None
        """
        return next((zs for zs in self.zs_lst if not zs.is_one_bi_zs()), None)

    def get_final_multi_bi_zs(self):
        """
        获取最后一个多笔中枢
        
        从中枢列表的末尾开始查找，返回最后一个多笔中枢
        
        Returns:
            CZS或None: 最后一个多笔中枢，如果不存在则返回None
        """
        zs_idx = len(self.zs_lst) - 1
        while zs_idx >= 0:
            zs = self.zs_lst[zs_idx]
            if not zs.is_one_bi_zs():
                return zs
            zs_idx -= 1
        return None

    def get_multi_bi_zs_cnt(self):
        """
        获取多笔中枢的数量
        
        统计线段中包含的多笔中枢总数，用于分析线段的复杂程度
        
        Returns:
            int: 多笔中枢的数量
        """
        return sum(not zs.is_one_bi_zs() for zs in self.zs_lst)
