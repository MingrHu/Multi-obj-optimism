"""Flask 路由 handler：只处理 HTTP 参数和响应，不承载业务逻辑。"""

from __future__ import annotations

from typing import Any, Callable

from flask import Blueprint, jsonify, request # type: ignore

from . import service
from .errors import ApiError

doe_api = Blueprint("doe_api", __name__)


#  @brief  生成统一成功响应
#  @return jsonify格式化信息和HTTP状态码 包含code message data
#  @param  data 接口返回的业务数据
#  @param  message 接口返回的提示信息
#  @param  status HTTP状态码
#  @author Hu Mingrui
#  @date   2026/08/25
def _ok(data: Any = None, message: str = "ok", status: int = 200):
    return jsonify({"code": 0, "message": message, "data": data if data is not None else {}}), status


#  @brief  解析POST请求的JSON请求体
#  @return dict格式请求参数
#  @param  request Flask当前请求对象 请求体必须为JSON对象
#  @author Hu Mingrui
#  @date   2026/08/25
def _body() -> dict[str, Any]:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ApiError("请求体必须是 JSON 对象")
    return value


#  @brief  调用service处理POST请求并生成统一成功响应
#  @return jsonify格式化信息和HTTP状态码 包含code message data
#  @param  action 接收JSON字典并返回业务数据的service处理函数
#  @param  message 请求成功后的提示信息
#  @param  status 请求成功后的HTTP状态码
#  @author Hu Mingrui
#  @date   2026/08/25
def _post(action: Callable[[dict[str, Any]], Any], message: str, status: int = 200):
    return _ok(action(_body()), message, status)


#  @brief  新建DOE任务
#  @return jsonify格式化信息和HTTP状态码 包含新建DOE任务的完整信息
#  @param  id DOE唯一标识 可选 未传入时由服务自动生成
#  @param  name DOE任务名称 可选
#  @param  description DOE任务描述 可选
#  @param  metadata DOE扩展信息JSON对象 可选
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.post("/api/v1/doe/add")
def add_doe():
    return _post(service.add_doe, "DOE 任务已创建", 201)


#  @brief  查询DOE任务列表
#  @return jsonify格式化信息 包含items任务列表和total任务总数
#  @param  无
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.get("/api/v1/doe/list")
def list_doe():
    tasks = service.list_doe()
    return _ok({"items": tasks, "total": len(tasks)})


#  @brief  删除DOE任务及全部关联文件
#  @return jsonify格式化信息 包含被删除的DOE唯一标识
#  @param  id DOE唯一标识 必填
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.post("/api/v1/doe/delete")
def delete_doe():
    return _post(service.delete_doe, "DOE 任务及相关文件已删除")


#  @brief  根据参数范围生成DOE样本
#  @return jsonify格式化信息 包含采样方法 参数范围和样本文件路径
#  @param  id DOE唯一标识 必填
#  @param  method 采样方法 支持lhs和full 默认为lhs
#  @param  param_ranges 参数范围JSON对象 每个参数对应lower和upper组成的数值数组
#  @param  n_samples LHS基础样本数量 使用lhs时必填
#  @param  level_nums 全因子各参数水平数量 使用full时必填
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.post("/api/v1/hust/doe/sample/generate")
def generate_sample():
    return _post(service.generate_sample, "样本生成完成")


#  @brief  在DOE目标目录下生成演示用代理模型训练数据集
#  @return jsonify格式化信息 包含数据文件路径 变量名称和样本数量
#  @param  id DOE唯一标识 必填
#  @param  param_ranges 输入参数范围JSON对象 必填
#  @param  target_names 目标变量名称列表 必填
#  @param  input_names 输入变量名称列表 可选 默认使用param_ranges的键顺序
#  @param  n_samples 训练样本数量 可选 默认100
#  @param  seed 随机种子 可选 默认42
#  @param  noise_ratio 目标值随机噪声比例 可选 默认0
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.post("/api/v1/hust/doe/dataset/generate")
def generate_training_dataset():
    return _post(service.generate_training_dataset, "训练数据集生成完成")


#  @brief  按字段获取样本 数据集 优化或推理结果
#  @return jsonify格式化信息 data仅包含请求字段到数据数组的映射
#  @param  id DOE唯一标识 必填
#  @param  data_type 数据类型 sample dataset optimization inference之一
#  @param  fields 需要返回的字段名称列表 必填 可重复传递
@doe_api.get("/api/v1/hust/doe/data/get")
def get_data():
    fields = request.args.getlist("fields")
    return _ok(service.get_data({
        "id": request.args.get("id", ""),
        "data_type": request.args.get("data_type", ""),
        "fields": fields,
    }), "数据获取完成")


#  @brief  查询DOE任务下代理模型的训练进度
#  @return jsonify格式化信息 包含训练状态 当前阶段 训练进度 已训练模型和错误信息
#  @param  id DOE唯一标识 GET查询参数 必填
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.get("/api/hust/v1/doe/train/progress")
def training_progress():
    return _ok(service.get_training_progress(request.args.get("id", "")))


#  @brief  删除DOE任务下的训练记录和代理模型文件
#  @return jsonify格式化信息 包含完成清理的DOE唯一标识
#  @param  id DOE唯一标识 必填
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.post("/api/v1/hust/doe/train/delete")
def delete_training():
    return _post(service.delete_training, "训练记录和模型文件已删除")


#  @brief  中止DOE任务下正在运行的代理模型训练
#  @return jsonify格式化信息 包含accepted中止请求是否被运行线程接受
#  @param  id DOE唯一标识 必填
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.post("/api/v1/hust/doe/train/stop")
def stop_training():
    accepted = service.stop_training(_body())
    return _ok({"accepted": accepted}, "已发送中止请求" if accepted else "没有运行中的训练")


#  @brief  提交代理模型训练和评价任务
#  @return jsonify格式化信息和HTTP 202状态码 包含已提交的DOE唯一标识
#  @param  id DOE唯一标识 必填
#  @param  data_file 训练数据文件路径 必填
#  @param  all_var_list 输入变量和目标变量名称列表 必填
#  @param  input_var_count 输入变量数量 必填
#  @param  models 待训练模型配置列表 可选 支持model_index和params
#  @param  evaluation 模型评价配置 可选 支持enabled n_splits random_state
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.post("/api/v1/hust/doe/train/startTrain")
def start_training():
    return _post(service.start_training, "训练任务已提交", 202)


#  @brief  加载DOE代理模型并执行批量推理
#  @return jsonify格式化信息 data仅包含请求目标字段到预测数组的映射
#  @param  id DOE唯一标识 必填
#  @param  inputs 输入参数二维数值数组 必填 单个样本可使用一维数组
#  @param  fields 需要返回的预测目标字段列表 可选 默认返回全部目标
#  @param  model_id 指定代理模型标识 可选 未传入时自动选择评分最高模型
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.post("/api/v1/hust/doe/inference/startInference")
def start_inference():
    return _post(service.start_inference, "推理完成")


#  @brief  提交单目标 多目标或强化学习优化任务
#  @return jsonify格式化信息和HTTP 202状态码 包含已提交的DOE唯一标识
#  @param  id DOE唯一标识 必填
#  @param  algorithm 优化算法 可选 支持nsga2 single single_objective multi_objective rl和reinforcement_learning
#  @param  model_id 指定代理模型标识 可选 未传入时自动选择评分最高模型
#  @param  objective_names 优化目标名称列表 NSGA2优化时必填
#  @param  all_var_list 输入变量和目标变量完整名称列表 NSGA2优化时必填
#  @param  input_var_count 输入变量数量 NSGA2优化时必填
#  @param  decision_var_names 决策变量名称列表 NSGA2优化时必填
#  @param  decision_var_indices 决策变量在输入变量中的下标列表 NSGA2优化时必填
#  @param  decision_bounds 决策变量上下界列表 NSGA2优化时必填
#  @param  objective_config 目标最小化或最大化配置列表 可选
#  @param  constraints 优化约束列表 可选
#  @param  optimizer_config 优化算法参数JSON对象 可选
#  @param  output_config 优化结果输出配置JSON对象 可选
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.post("/api/v1/hust/doe/optimize/start")
def start_optimization():
    return _post(service.start_optimization, "优化任务已提交", 202)


#  @brief  中止DOE任务下正在运行的优化任务
#  @return jsonify格式化信息 包含accepted中止请求是否被运行线程接受
#  @param  id DOE唯一标识 必填
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.post("/api/v1/hust/doe/optimize/stop")
def stop_optimization():
    accepted = service.stop_optimization(_body())
    return _ok({"accepted": accepted}, "已发送中止请求" if accepted else "没有运行中的优化")


#  @brief  根据DOE唯一标识查询优化状态和结果
#  @return jsonify格式化信息 包含优化状态 请求参数 错误信息和结果文件
#  @param  id DOE唯一标识 GET查询参数 必填
#  @author Hu Mingrui
#  @date   2026/08/25
@doe_api.get("/api/v1/hust/doe/optimize/getById")
def get_optimization():
    return _ok(service.get_optimization(request.args.get("id", "")))


#  @brief  注册API统一异常处理函数
#  @return None
#  @param  app Flask应用实例
#  @author Hu Mingrui
#  @date   2026/08/25
def register_error_handlers(app) -> None:
    #  @brief  将业务异常转换为指定业务码和HTTP状态码
    #  @return jsonify格式化错误信息和HTTP状态码
    #  @param  error ApiError业务异常
    #  @author Hu Mingrui
    #  @date   2026/08/25
    @app.errorhandler(ApiError)
    def handle_api_error(error):
        return jsonify({"code": error.code, "message": error.message, "data": {}}), error.status

    #  @brief  将基础参数异常转换为HTTP 400响应
    #  @return jsonify格式化错误信息和HTTP 400状态码
    #  @param  error ValueError参数异常
    #  @author Hu Mingrui
    #  @date   2026/08/25
    @app.errorhandler(ValueError)
    def handle_value_error(error):
        return jsonify({"code": 1, "message": str(error), "data": {}}), 400

    #  @brief  记录未预期异常并转换为HTTP 500响应
    #  @return jsonify格式化错误信息和HTTP 500状态码
    #  @param  error 未预期异常
    #  @author Hu Mingrui
    #  @date   2026/08/25
    @app.errorhandler(Exception)
    def handle_unexpected(error):
        app.logger.exception("Unhandled API error")
        return jsonify({"code": 500, "message": "服务内部错误", "data": {}}), 500


__all__ = ["doe_api", "register_error_handlers"]
