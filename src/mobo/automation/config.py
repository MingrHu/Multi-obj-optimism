"""DEFORM 平台配置。

从 ``AutoScript/utils.py`` 搬入的 :class:`DeformConfig`，统一管理 KEY 文件关键字、
模拟对象定义与目标函数映射。其目标函数（``TAR_FUNC``）复用原子能力层中的提取
函数（:mod:`mobo.extraction.deform_targets`），仅调整 import 来源，映射内容与
行为保持不变，供 :func:`mobo.automation.extract.extract_dataset` 调用。
"""

from typing import Any

from mobo.extraction.deform_targets import (
    _extractGrainStdv,
    _extractMaxLoad,
    _extractMaxStress,
)


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
        'pressure_roll_speed_low':"MOVCTL",# 碾环驱动辊的函数速率最小值

        'temp': "NDTMP",
        'speed': "MOVCTL",
    }

    # ===================== 模拟对象定义 =====================
    OBJ_DEF = {
        'workpiece': "1",
        'driving_roll':"2",
        'pressure_roll':"3",
        'axial_roll_1':"4",
        'axial_roll_2':"5",

        'topdie': "2",
        'butdie': "3"
    }

    # ===================== 目标函数映射 =====================
    # 注意：_extractMaxStress 等函数需要在类定义前已声明
    TAR_FUNC: dict[str, Any] = {
        'stress': _extractMaxStress,
        'load': _extractMaxLoad,
        'grain': _extractGrainStdv
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
        安全获取目标函数
        :param func_name: stress/load/grain
        :return: 对应提取函数
        """
        return cls.TAR_FUNC.get(func_name)
