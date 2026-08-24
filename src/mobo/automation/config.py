"""DEFORM 平台配置。
从 ``AutoScript/utils.py`` 搬入的 :class:`DeformConfig`，统一管理 KEY 文件关键字、
模拟对象定义与目标函数映射
"""

from functools import partial
from typing import Any, List

from mobo.extraction.deform_targets import (
    _extractEffectiveStrainStdv,
    _extractGrainMorph,
    _extractMaxLoad,
    _extractMaxStress,
    _extractMaterialFill,
    _extractUsrGrainStdv,
)
from mobo.extraction.ring_roundness import extract_ring_roundness


# ===================== 目标提取适配器 =====================
def _lines_target(extract_fn, key_files: List[str], frames: List[List[str]],
                  obj_id: str, in_progress: bool, select_component=None) -> str:
    """key_lines 约定：只用逐帧文本行计算（应力/载荷/晶粒）。"""
    return extract_fn(frames, obj_id, in_progress, select_component)


def _roundness_target(which: str, key_files: List[str], frames: List[List[str]],
                      obj_id: str, in_progress: bool, select_component=None) -> str:
    """key_file 约定：用最终步 KEY 文件几何计算内/外圈圆度。"""
    value = extract_ring_roundness(key_files[-1], which=which, object_id=int(obj_id))
    return "{:.6f}".format(value)


##########################################################
# *********************SOME VAR DEF***********************
# 针对DEFORM平台的一些Config定义
# 调用方式:
# temp_key = DeformConfig.get_key_var('temp')
# workpiece_id = DeformConfig.get_object_id('workpiece')
# stress_func = DeformConfig.get_target_function('stress')
class DeformConfig:
    """
    DEFORM 平台配置类
    统一管理关键字、对象定义、目标函数等配置信息
    """

    # ===================== KEY文件关键字变量 =====================
    KEYFILE_VAR_DIC = {
        'roll_tmp':"REFTMP",    # 碾环锻造的温度字典
        'pressure_roll_speed_upper':"MOVCTL", # 碾环驱动辊的函数速率最大值
        'pressure_roll_speed_lower':"MOVCTL",# 碾环驱动辊的函数速率最小值
        'driving_roll_rad_speed':"ANGMOV", # 驱动辊的角速度

        'temp': "NDTMP",
        'speed': "MOVCTL",
    }

    # ===================== 模拟对象定义 =====================
    OBJ_DEF = {
        'workpiece': "1",       # 普通工件
        'driving_roll':"2",     # 碾环驱动辊
        'pressure_roll':"3",    # 碾环压力辊
        'axial_roll_1':"4",     # 定位辊1
        'axial_roll_2':"5",     # 定位辊2

        'topdie': "2",     # 普通压力锻造上模
        'butdie': "3"      # 普通压力锻造下模
    }

    # ===================== 目标函数映射 =====================
    # 统一签名 fn(key_files, frames, obj_id, in_progress, select_component) -> str
    TAR_FUNC: dict[str, Any] = {
        'stress': partial(_lines_target, _extractMaxStress),
        'load': partial(_lines_target, _extractMaxLoad),
        'strain_std': partial(_lines_target, _extractEffectiveStrainStdv),
        'grain': partial(_lines_target, _extractUsrGrainStdv),
        'grain_morph': partial(_lines_target, _extractGrainMorph),
        'material_fill': partial(_lines_target, _extractMaterialFill),
        'roundness_inner': partial(_roundness_target, "inner"),
        'roundness_outer': partial(_roundness_target, "outer"),
    }

    @classmethod
    def get_key_var(cls, key: str):
        """
        安全获取 KEY 文件关键字
        :param key: 配置键名 temp/speed
        :return: 对应关键字
        """
        return cls.KEYFILE_VAR_DIC.get(key)

    @classmethod
    def get_object_id(cls, obj_name: str):
        """
        安全获取对象 ID
        :param obj_name: workpiece/topdie/butdie
        :return: 对象ID字符串
        """
        return cls.OBJ_DEF.get(obj_name)

    @classmethod
    def get_target_function(cls, func_name: str)->Any:
        """
        安全获取目标函数（统一签名 fn(key_files, frames, obj_id, in_progress)）
        :param func_name: stress/load/grain/roundness_inner/roundness_outer
        :return: 对应提取适配器
        """
        return cls.TAR_FUNC.get(func_name)
