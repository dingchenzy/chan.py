"""
缠论配置模块

本模块定义了缠论分析的各种配置参数，包括笔、线段、中枢、买卖点等相关配置。
通过这些配置可以调整缠论分析的各种行为和参数。
"""

from typing import List

from Bi.BiConfig import CBiConfig
from BuySellPoint.BSPointConfig import CBSPointConfig
from Common.CEnum import TREND_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.func_util import _parse_inf  # 导入无穷大值解析函数
from Math.BOLL import BollModel  # 布林带模型
from Math.Demark import CDemarkEngine  # Demark指标引擎
from Math.KDJ import KDJ  # KDJ指标
from Math.MACD import CMACD  # MACD指标
from Math.RSI import RSI  # RSI指标
from Math.TrendModel import CTrendModel  # 趋势模型
from Seg.SegConfig import CSegConfig  # 线段配置
from ZS.ZSConfig import CZSConfig  # 中枢配置


class CChanConfig:
    """
    缠论配置类
    
    该类包含了缠论分析的所有配置参数，包括笔、线段、中枢、买卖点等相关配置。
    """
    def __init__(self, conf=None):
        """
        初始化缠论配置
        
        Args:
            conf: 配置字典，默认为None
        """
        if conf is None:
            conf = {}
        conf = ConfigWithCheck(conf)
        
        # 笔配置
        self.bi_conf = CBiConfig(
            bi_algo=conf.get("bi_algo", "normal"),          # 笔算法，默认为normal (normal:按缠论笔定义来算, fx:顶底分形即成笔)
            is_strict=conf.get("bi_strict", True),          # 是否严格模式 (bi_algo=normal时有效)
            bi_fx_check=conf.get("bi_fx_check", "strict"),  # 分型检查模式 (strict:底分型最低点必须比顶分型3元素最低点最小值还低，顶分型反之; totally:底分型3元素最高点必须比顶分型三元素最低点还低; loss:底分型最低点比顶分型中间元素低点还低，顶分型反之; half:对于上升笔，底分型最低点比顶分型前两元素最低点还低，顶分型最高点比底分型后两元素高点还高，下降笔反之)
            gap_as_kl=conf.get("gap_as_kl", False),         # 是否将跳空视为K线
            bi_end_is_peak=conf.get('bi_end_is_peak', True), # 笔的尾部是否是整笔中最低/最高
            bi_allow_sub_peak=conf.get("bi_allow_sub_peak", True), # 是否允许次高点成笔
        )
        
        # 线段配置
        self.seg_conf = CSegConfig(
            seg_algo=conf.get("seg_algo", "chan"),          # 线段算法，默认为chan (chan:利用特征序列来计算, 1+1:都业华版本1+1终结算法, break:线段破坏定义来计算线段)
            left_method=conf.get("left_seg_method", "peak"), # 左侧方法 (all:收集至最后一个方向正确的笔成为一段, peak:如果有个靠谱的新的极值，那么分成两段)
        )
        
        # 中枢配置
        self.zs_conf = CZSConfig(
            need_combine=conf.get("zs_combine", True),      # 是否需要合并中枢
            zs_combine_mode=conf.get("zs_combine_mode", "zs"), # 中枢合并模式 (zs:两中枢区间有重叠才合并, peak:两中枢有K线重叠就合并)
            one_bi_zs=conf.get("one_bi_zs", False),         # 是否允许一笔中枢 (分析趋势时会用到)
            zs_algo=conf.get("zs_algo", "normal"),          # 中枢算法 (normal:段内中枢, over_seg:跨段中枢, auto:自动)
        )

        # 步进模式配置
        self.trigger_step = conf.get("trigger_step", False) # 是否触发步进模式 (用于逐步回放绘图，CChan会变成生成器，每读取一根新K线就会计算一次当前所有指标)
        self.skip_step = conf.get("skip_step", 0)           # 跳过步数 (trigger_step为True时有效，指定跳过前面几根K线)

        # K线数据检查配置
        self.kl_data_check = conf.get("kl_data_check", True) # 是否检查K线数据 (检查时间线是否有乱序，大小级别K线是否有缺失)
        self.max_kl_misalgin_cnt = conf.get("max_kl_misalgin_cnt", 2) # 最大K线不对齐计数 (在次级别找不到K线最大条数，次级别数据有缺失)
        self.max_kl_inconsistent_cnt = conf.get("max_kl_inconsistent_cnt", 5) # 最大K线不一致计数 (天K线以下子级别和父级别日期不一致最大允许条数)
        self.auto_skip_illegal_sub_lv = conf.get("auto_skip_illegal_sub_lv", False) # 是否自动跳过非法子级别 (如果获取次级别数据失败，自动删除该级别)
        self.print_warning = conf.get("print_warning", True) # 是否打印警告 (打印K线不一致的明细)
        self.print_err_time = conf.get("print_err_time", True) # 是否打印错误时间 (计算发生错误时打印因为什么时间的K线数据导致的)

        # 指标配置
        self.mean_metrics: List[int] = conf.get("mean_metrics", []) # 均线周期列表 (用于生成特征及绘图时使用，例如[5,20])
        self.trend_metrics: List[int] = conf.get("trend_metrics", []) # 趋势线周期列表 (计算上下轨道线周期，即T天内最高/低价格)
        self.macd_config = conf.get("macd", {"fast": 12, "slow": 26, "signal": 9}) # MACD配置
        self.cal_demark = conf.get("cal_demark", False)     # 是否计算Demark指标
        self.cal_rsi = conf.get("cal_rsi", False)           # 是否计算RSI指标
        self.cal_kdj = conf.get("cal_kdj", False)           # 是否计算KDJ指标
        self.rsi_cycle = conf.get("rsi_cycle", 14)          # RSI周期
        self.kdj_cycle = conf.get("kdj_cycle", 9)           # KDJ周期
        
        # Demark指标配置
        self.demark_config = conf.get("demark", {
            'demark_len': 9,                # Demark长度 (setup完成时长度)
            'setup_bias': 4,                # 设置偏差 (setup比较偏移量)
            'countdown_bias': 2,            # 倒计时偏差 (countdown比较偏移量)
            'max_countdown': 13,            # 最大倒计时数
            'tiaokong_st': True,            # 跳空设置 (序列真实起始位置计算时，如果setup第一根跳空，是否需要取前一根收盘价)
            'setup_cmp2close': True,        # 设置比较收盘价 (setup计算当前K线的收盘价对比的是setup_bias根K线前的close，如果不是，下跌setup对比的是low，上升对比的是close)
            'countdown_cmp2close': True,    # 倒计时比较收盘价 (countdown计算当前K线的收盘价对比的是countdown_bias根K线前的close，如果不是，下跌setup对比的是low，上升对比的是close)
        })
        
        self.boll_n = conf.get("boll_n", 20)  # 布林带周期 (布林线参数N)
        
        # 设置买卖点配置
        self.set_bsp_config(conf)

        # 检查配置是否有未使用的参数
        conf.check()

    def GetMetricModel(self):
        """
        获取所有配置的指标模型
        
        Returns:
            List: 返回所有配置的指标模型列表，包括MACD、均线、趋势线、布林带、Demark、RSI、KDJ等
        """
        res: List[CMACD | CTrendModel | BollModel | CDemarkEngine | RSI | KDJ] = [
            CMACD(
                fastperiod=self.macd_config['fast'],
                slowperiod=self.macd_config['slow'],
                signalperiod=self.macd_config['signal'],
            )
        ]
        res.extend(CTrendModel(TREND_TYPE.MEAN, mean_T) for mean_T in self.mean_metrics)

        for trend_T in self.trend_metrics:
            res.append(CTrendModel(TREND_TYPE.MAX, trend_T))
            res.append(CTrendModel(TREND_TYPE.MIN, trend_T))
        res.append(BollModel(self.boll_n))
        if self.cal_demark:
            res.append(CDemarkEngine(
                demark_len=self.demark_config['demark_len'],
                setup_bias=self.demark_config['setup_bias'],
                countdown_bias=self.demark_config['countdown_bias'],
                max_countdown=self.demark_config['max_countdown'],
                tiaokong_st=self.demark_config['tiaokong_st'],
                setup_cmp2close=self.demark_config['setup_cmp2close'],
                countdown_cmp2close=self.demark_config['countdown_cmp2close'],
            ))
        if self.cal_rsi:
            res.append(RSI(self.rsi_cycle))
        if self.cal_kdj:
            res.append(KDJ(self.kdj_cycle))
        return res

    def set_bsp_config(self, conf):
        """
        设置买卖点配置
        
        Args:
            conf: 配置字典，包含买卖点相关的配置参数
        """
        para_dict = {
            "divergence_rate": float("inf"),    # 背驰度，默认为无穷大
            "min_zs_cnt": 1,                  # 最小中枢数量
            "bsp1_only_multibi_zs": True,     # 一类买卖点是否只考虑多笔中枢
            "max_bs2_rate": 0.9999,           # 最大二类买卖点比率
            "macd_algo": "peak",              # MACD算法，默认为peak
            "bs1_peak": True,                # 一类买卖点是否必须是顶底分型
            "bs_type": "1,1p,2,2s,3a,3b",     # 买卖点类型
            "bsp2_follow_1": True,            # 二类买卖点是否跟随一类买卖点
            "bsp3_follow_1": True,            # 三类买卖点是否跟随一类买卖点
            "bsp3_peak": False,               # 三类买卖点是否必须是顶底分型
            "bsp2s_follow_2": False,          # 特殊二类买卖点是否跟随二类买卖点
            "max_bsp2s_lv": None,             # 最大特殊二类买卖点级别
            "strict_bsp3": False,             # 是否严格三类买卖点
        }
        args = {para: conf.get(para, default_value) for para, default_value in para_dict.items()}
        self.bs_point_conf = CBSPointConfig(**args)

        # 线段买卖点配置，基于普通买卖点配置但有特殊设置
        self.seg_bs_point_conf = CBSPointConfig(**args)
        self.seg_bs_point_conf.b_conf.set("macd_algo", "slope")  # 线段买点使用斜率算法
        self.seg_bs_point_conf.s_conf.set("macd_algo", "slope")  # 线段卖点使用斜率算法
        self.seg_bs_point_conf.b_conf.set("bsp1_only_multibi_zs", False)  # 线段一类买点不限制多笔中枢
        self.seg_bs_point_conf.s_conf.set("bsp1_only_multibi_zs", False)  # 线段一类卖点不限制多笔中枢

        # 处理特殊配置参数
        for k, v in conf.items():
            if isinstance(v, str):
                v = f'"{v}"'
            v = _parse_inf(v)
            if k.endswith("-buy"):
                prop = k.replace("-buy", "")
                exec(f"self.bs_point_conf.b_conf.set('{prop}', {v})")
            elif k.endswith("-sell"):  # 卖点特殊配置
                prop = k.replace("-sell", "")
                exec(f"self.bs_point_conf.s_conf.set('{prop}', {v})")
            elif k.endswith("-segbuy"):  # 线段买点特殊配置
                prop = k.replace("-segbuy", "")
                exec(f"self.seg_bs_point_conf.b_conf.set('{prop}', {v})")
            elif k.endswith("-segsell"):  # 线段卖点特殊配置
                prop = k.replace("-segsell", "")
                exec(f"self.seg_bs_point_conf.s_conf.set('{prop}', {v})")
            elif k.endswith("-seg"):  # 线段买卖点通用配置
                prop = k.replace("-seg", "")
                exec(f"self.seg_bs_point_conf.b_conf.set('{prop}', {v})")
                exec(f"self.seg_bs_point_conf.s_conf.set('{prop}', {v})")
            elif k in args:  # 通用买卖点配置
                exec(f"self.bs_point_conf.b_conf.set({k}, {v})")
                exec(f"self.bs_point_conf.s_conf.set({k}, {v})")
            else:
                raise CChanException(f"unknown para = {k}", ErrCode.PARA_ERROR)  # 未知参数错误
        
        # 解析目标类型
        self.bs_point_conf.b_conf.parse_target_type()
        self.bs_point_conf.s_conf.parse_target_type()
        self.seg_bs_point_conf.b_conf.parse_target_type()
        self.seg_bs_point_conf.s_conf.parse_target_type()


class ConfigWithCheck:
    """
    带检查功能的配置类
    
    该类用于跟踪配置参数的使用情况，确保所有提供的配置参数都被正确使用。
    """
    def __init__(self, conf):
        """
        初始化配置检查类
        
        Args:
            conf: 配置字典
        """
        self.conf = conf

    def get(self, k, default_value=None):
        """
        获取配置参数值并标记为已使用
        
        Args:
            k: 配置参数名
            default_value: 默认值，当参数不存在时返回
            
        Returns:
            配置参数值或默认值
        """
        res = self.conf.get(k, default_value)
        if k in self.conf:
            del self.conf[k]
        return res

    def items(self):
        """
        遍历所有配置项并标记为已使用
        
        Yields:
            配置参数的键值对
        """
        visit_keys = set()
        for k, v in self.conf.items():
            yield k, v
            visit_keys.add(k)
        for k in visit_keys:
            del self.conf[k]

    def check(self):
        """
        检查是否所有配置参数都已被使用
        
        如果有未使用的参数，将抛出异常
        
        Raises:
            CChanException: 当有未使用的配置参数时
        """
        if len(self.conf) > 0:
            invalid_key_lst = ",".join(list(self.conf.keys()))
            raise CChanException(f"invalid CChanConfig: {invalid_key_lst}", ErrCode.PARA_ERROR)
