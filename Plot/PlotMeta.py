from typing import List

from Bi.Bi import CBi
from BuySellPoint.BS_Point import CBS_Point
from Common.CEnum import FX_TYPE
from KLine.KLine import CKLine
from KLine.KLine_List import CKLine_List
from Seg.Eigen import CEigen
from Seg.EigenFX import CEigenFX
from Seg.Seg import CSeg
from ZS.ZS import CZS


class Cklc_meta:
    """
    K线组合（CKLine）的元数据封装类。
    用于将原始 K 线组合对象中绘图所需的关键字段提取出来，方便后续绘图模块使用。
    """
    def __init__(self, klc: CKLine):
        # 最高价与最低价
        self.high = klc.high
        self.low = klc.low
        # 该 K 线组合在序列中的起始与结束索引
        self.begin_idx = klc.lst[0].idx
        self.end_idx = klc.lst[-1].idx
        # 分型类型，若未知则使用方向替代
        self.type = klc.fx if klc.fx != FX_TYPE.UNKNOWN else klc.dir

        # 包含的所有 K 线单元列表
        self.klu_list = list(klc.lst)


class CBi_meta:
    """
    笔（CBi）的元数据封装类。
    提取笔的方向、端点坐标、确定性等绘图所需信息。
    """
    def __init__(self, bi: CBi):
        # 笔的序号
        self.idx = bi.idx
        # 方向：向上/向下
        self.dir = bi.dir
        # 笔的类型
        self.type = bi.type
        # 笔起点与终点在 K 线序列中的索引
        self.begin_x = bi.get_begin_klu().idx
        self.end_x = bi.get_end_klu().idx
        # 笔起点与终点的价格
        self.begin_y = bi.get_begin_val()
        self.end_y = bi.get_end_val()
        # 是否为确定笔
        self.is_sure = bi.is_sure


class CSeg_meta:
    """
    线段（CSeg）的元数据封装类。
    提取线段端点、方向、趋势线等信息。
    """
    def __init__(self, seg: CSeg):
        # 线段端点坐标提取：若线段直接由笔组成，则直接取笔端点；
        # 若线段由子线段组成，则递归取子线段的笔端点。
        if isinstance(seg.start_bi, CBi):
            self.begin_x = seg.start_bi.get_begin_klu().idx
            self.begin_y = seg.start_bi.get_begin_val()
            self.end_x = seg.end_bi.get_end_klu().idx
            self.end_y = seg.end_bi.get_end_val()
        else:
            assert isinstance(seg.start_bi, CSeg)
            self.begin_x = seg.start_bi.start_bi.get_begin_klu().idx
            self.begin_y = seg.start_bi.start_bi.get_begin_val()
            self.end_x = seg.end_bi.end_bi.get_end_klu().idx
            self.end_y = seg.end_bi.end_bi.get_end_val()
        # 线段方向
        self.dir = seg.dir
        # 是否确定
        self.is_sure = seg.is_sure
        # 线段序号
        self.idx = seg.idx

        # 趋势线字典：支持/阻力
        self.tl = {}
        if seg.support_trend_line and seg.support_trend_line.line:
            self.tl["support"] = seg.support_trend_line
        if seg.resistance_trend_line and seg.resistance_trend_line.line:
            self.tl["resistance"] = seg.resistance_trend_line

    def format_tl(self, tl):
        """
        将趋势线对象转换为绘图可用的起点与终点坐标。
        参数
        ----
        tl : 趋势线对象，必须包含有效的 line 属性。

        返回
        ----
        (x0, y0, x1, y1) : 趋势线的起点与终点坐标。
        """
        assert tl.line
        # 防止除零，给斜率加极小值
        tl_slope = tl.line.slope + 1e-7
        tl_x = tl.line.p.x
        tl_y = tl.line.p.y
        tl_y0 = self.begin_y
        tl_y1 = self.end_y
        # 根据直线方程反推 x 坐标
        tl_x0 = (tl_y0-tl_y)/tl_slope + tl_x
        tl_x1 = (tl_y1-tl_y)/tl_slope + tl_x
        return tl_x0, tl_y0, tl_x1, tl_y1


class CEigen_meta:
    """
    特征序列元素（CEigen）的元数据封装类。
    提取高低点、宽度、高度等绘图信息。
    """
    def __init__(self, eigen: CEigen):
        # 元素在 K 线序列中的起止索引
        self.begin_x = eigen.lst[0].get_begin_klu().idx
        self.end_x = eigen.lst[-1].get_end_klu().idx
        # 元素的高低价格
        self.begin_y = eigen.low
        self.end_y = eigen.high
        # 宽度与高度
        self.w = self.end_x - self.begin_x
        self.h = self.end_y - self.begin_y


class CEigenFX_meta:
    """
    特征序列分型（CEigenFX）的元数据封装类。
    内部包含三个特征元素，中间元素为分型核心。
    """
    def __init__(self, eigenFX: CEigenFX):
        # 将三个元素（允许空）转换为元数据对象
        self.ele = [CEigen_meta(ele) for ele in eigenFX.ele if ele is not None]
        assert len(self.ele) == 3
        assert eigenFX.ele[1] is not None
        # 中间元素的缺口与分型类型
        self.gap = eigenFX.ele[1].gap
        self.fx = eigenFX.ele[1].fx


class CZS_meta:
    """
    中枢（CZS）的元数据封装类。
    提取中枢高低、区间、子中枢等信息。
    """
    def __init__(self, zs: CZS):
        # 中枢高低价格
        self.low = zs.low
        self.high = zs.high
        # 中枢在序列中的起止索引
        self.begin = zs.begin.idx
        self.end = zs.end.idx
        # 中枢宽度与高度
        self.w = self.end - self.begin
        self.h = self.high - self.low
        # 是否确定
        self.is_sure = zs.is_sure
        # 递归封装子中枢
        self.sub_zs_lst = [CZS_meta(t) for t in zs.sub_zs_lst]
        # 是否为一线中枢
        self.is_onebi_zs = zs.is_one_bi_zs()


class CBS_Point_meta:
    """
    买卖点（CBS_Point）的元数据封装类。
    记录买卖方向、类型、坐标及是否属于线段买卖点。
    """
    def __init__(self, bsp: CBS_Point, is_seg):
        # 买卖方向
        self.is_buy = bsp.is_buy
        # 类型字符串
        self.type = bsp.type2str()
        # 是否线段买卖点
        self.is_seg = is_seg

        # 买卖点所在 K 线索引
        self.x = bsp.klu.idx
        # 买卖点价格：买取低，卖取高
        self.y = bsp.klu.low if self.is_buy else bsp.klu.high

    def desc(self):
        """
        返回买卖点的简短描述字符串。
        线段买卖点前加“※”前缀。
        """
        is_seg_flag = "※" if self.is_seg else ""
        return f'{is_seg_flag}b{self.type}' if self.is_buy else f'{is_seg_flag}s{self.type}'


class CChanPlotMeta:
    """
    缠论绘图所需全部元数据的容器类。
    统一封装 K 线、笔、线段、中枢、买卖点等信息，供绘图模块直接调用。
    """
    def __init__(self, kl_list: CKLine_List):
        # 原始 K 线列表对象
        self.data = kl_list

        # K 线组合元数据列表
        self.klc_list: List[Cklc_meta] = [Cklc_meta(klc) for klc in kl_list.lst]
        # 时间字符串序列，用于横轴刻度
        self.datetick = [klu.time.to_str() for klu in self.klu_iter()]
        # K 线单元总数量
        self.klu_len = sum(len(klc.klu_list) for klc in self.klc_list)

        # 笔元数据列表
        self.bi_list = [CBi_meta(bi) for bi in kl_list.bi_list]

        # 线段元数据列表 & 对应特征序列分型
        self.seg_list: List[CSeg_meta] = []
        self.eigenfx_lst: List[CEigenFX_meta] = []
        for seg in kl_list.seg_list:
            self.seg_list.append(CSeg_meta(seg))
            if seg.eigen_fx:
                self.eigenfx_lst.append(CEigenFX_meta(seg.eigen_fx))

        # 子线段元数据列表 & 对应特征序列分型
        self.seg_eigenfx_lst: List[CEigenFX_meta] = []
        self.segseg_list: List[CSeg_meta] = []
        for segseg in kl_list.segseg_list:
            self.segseg_list.append(CSeg_meta(segseg))
            if segseg.eigen_fx:
                self.seg_eigenfx_lst.append(CEigenFX_meta(segseg.eigen_fx))

        # 中枢元数据列表（笔中枢 & 线段中枢）
        self.zs_lst: List[CZS_meta] = [CZS_meta(zs) for zs in kl_list.zs_list]
        self.segzs_lst: List[CZS_meta] = [CZS_meta(segzs) for segzs in kl_list.segzs_list]

        # 买卖点元数据列表（笔买卖点 & 线段买卖点）
        self.bs_point_lst: List[CBS_Point_meta] = [CBS_Point_meta(bs_point, is_seg=False) for bs_point in kl_list.bs_point_lst.bsp_iter()]
        self.seg_bsp_lst: List[CBS_Point_meta] = [CBS_Point_meta(seg_bsp, is_seg=True) for seg_bsp in kl_list.seg_bs_point_lst.bsp_iter()]

    def klu_iter(self):
        """
        生成器：按顺序遍历所有 K 线单元。
        """
        for klc in self.klc_list:
            yield from klc.klu_list

    def sub_last_kseg_start_idx(self, seg_cnt):
        """
        获取倒数第 seg_cnt 个线段在子 K 线中的起始索引。
        若 seg_cnt 过大或为空，则返回 0。
        """
        if seg_cnt is None or len(self.data.seg_list) <= seg_cnt:
            return 0
        else:
            return self.data.seg_list[-seg_cnt].get_begin_klu().sub_kl_list[0].idx

    def sub_last_kbi_start_idx(self, bi_cnt):
        """
        获取倒数第 bi_cnt 个笔在子 K 线中的起始索引。
        若 bi_cnt 过大或为空，则返回 0。
        """
        if bi_cnt is None or len(self.data.bi_list) <= bi_cnt:
            return 0
        else:
            return self.data.bi_list[-bi_cnt].begin_klc.lst[0].sub_kl_list[0].idx

    def sub_range_start_idx(self, x_range):
        """
        从后往前数 x_range 根 K 线，返回其在子 K 线中的起始索引。
        用于局部缩放绘图。
        """
        for klc in self.data[::-1]:
            for klu in klc[::-1]:
                x_range -= 1
                if x_range == 0:
                    return klu.sub_kl_list[0].idx
        return 0
