# 优化与代理模型模块开源依赖合规清单

本文档用于国产自主软件优化与代理模型模块的交付沟通与合规登记，范围为本仓库中
优化模块、代理模型模块及其 API 服务实际使用的主要第三方依赖。

注意：

- 本文作为后端服务开源依赖登记文件使用。正式商用交付以实际发布包中的 `METADATA`、
  `LICENSE`、`NOTICE` 文件及法务审查结论为准。
- 下表覆盖后端服务的直接运行依赖与生产部署依赖；`requirements/runtime.txt`
  中的传递依赖较多，正式发行环境应使用 SBOM 或 `pip-licenses` 生成完整清单。
## 运行时直接依赖

| 开源库 | 在本项目中的作用 | 下载网址 | 开源协议 | 是否有侵权影响/合规注意 |
|---|---|---|---|---|
| NumPy | 数值数组、矩阵计算、模型输入输出与优化变量计算基础能力 | https://pypi.org/project/numpy/ | BSD-3-Clause | 宽松许可证。按要求保留版权和许可证声明后，可用于闭源商业软件。 |
| pandas | TSV/表格数据读写、训练数据集与优化结果数据处理 | https://pypi.org/project/pandas/ | BSD-3-Clause | 宽松许可证。按要求保留版权和许可证声明后，可用于闭源商业软件。 |
| SciPy | 科学计算、优化/统计相关基础能力 | https://pypi.org/project/scipy/ | BSD-3-Clause | 宽松许可证。按要求保留版权和许可证声明后，可用于闭源商业软件。 |
| scikit-learn | 代理模型训练与评估：随机森林、SVR、Kriging/GPR、标准化、交叉验证等 | https://pypi.org/project/scikit-learn/ | BSD-3-Clause | 宽松许可证。按要求保留版权和许可证声明后，可用于闭源商业软件。 |
| matplotlib | Pareto 前沿、训练/优化结果图形输出 | https://pypi.org/project/matplotlib/ | Matplotlib License / PSF-style | 宽松许可证。按要求保留版权和许可证声明后，可用于闭源商业软件。 |
| joblib | 模型、标准化器等对象序列化与加载 | https://pypi.org/project/joblib/ | BSD-3-Clause | 宽松许可证。按要求保留版权和许可证声明后，可用于闭源商业软件。 |
| Keras | DNN 代理模型搭建与训练接口 | https://pypi.org/project/keras/ | Apache-2.0 | 允许商用/闭源集成。需保留许可证、NOTICE；注意 Apache-2.0 的专利授权与终止条款。 |
| TensorFlow | Keras 后端与深度学习运行时 | https://pypi.org/project/tensorflow/ | Apache-2.0 | 允许商用/闭源集成。需保留许可证、NOTICE；发行包较大且含传递依赖，发行时需生成完整第三方声明。 |
| pymoo | NSGA-II 多目标遗传优化算法与可视化 | https://pypi.org/project/pymoo/ | Apache-2.0 | 允许商用/闭源集成。需保留许可证、NOTICE；算法名称引用应避免暗示官方背书。 |
| Gymnasium | 强化学习环境接口，供优化环境封装使用 | https://pypi.org/project/gymnasium/ | MIT | 宽松许可证。按要求保留版权和许可证声明后，可用于闭源商业软件。 |
| stable-baselines3 | PPO 强化学习训练与优化 | https://pypi.org/project/stable-baselines3/ | MIT | 宽松许可证。按要求保留版权和许可证声明后，可用于闭源商业软件。 |
| PyTorch / torch | 强化学习底层张量计算；stable-baselines3 训练依赖 | https://pypi.org/project/torch/ | BSD-style | 宽松许可证。按要求保留版权和许可证声明后，可用于闭源商业软件；本项目推荐使用 CPU wheel 源安装。 |
| Flask | DOE HTTP API 服务框架 | https://pypi.org/project/Flask/ | BSD-3-Clause | 宽松许可证。按要求保留版权和许可证声明后，可用于闭源商业软件。 |
| requests | API demo/客户端请求调用 | https://pypi.org/project/requests/ | Apache-2.0 | 允许商用/闭源集成。需保留许可证、NOTICE。 |

## 服务部署依赖

| 开源库 | 在本项目中的作用 | 下载网址 | 开源协议 | 是否有侵权影响/合规注意 |
|---|---|---|---|---|
| Gunicorn | Linux 容器/生产环境运行 Flask API 的 WSGI 服务 | https://pypi.org/project/gunicorn/ | MIT | 宽松许可证。按要求保留版权和许可证声明后，可用于闭源商业软件。仅 Linux/类 Unix 部署需要。 |

## 合规结论

1. 优化与代理模型模块主要运行依赖采用 BSD、MIT 或 Apache-2.0 等宽松许可证。按要求保留版权声明、
   许可证文本和 NOTICE 后，可用于闭源商业软件。
2. 依赖组合覆盖数值计算、训练数据处理、代理模型训练、模型持久化、多目标优化、强化学习优化、
   HTTP API 服务和生产部署能力，能够支撑优化与代理模型模块的功能需求。
3. 当前优化与代理模型模块依赖不存在 GPL-only 或强制开源自研代码的许可证。按许可证要求完成声明与
   随包文件交付后，不构成开源许可证侵权风险。
4. 本仓库自身在 `pyproject.toml` 中声明为 `Proprietary`。应确保自研代码与第三方库边界清晰，
   不把 GPL-only 代码复制进本仓库源码。
5. 面向客户交付 Docker 镜像、离线安装包或服务部署包时，应随包提供
   `THIRD_PARTY_NOTICES`、许可证文本、版本号、下载来源和修改说明。
6. 正式发布前应生成第三方依赖清单：

   ```bash
   python -m pip install pip-licenses
   pip-licenses --format=markdown --with-urls --with-license-file --output-file THIRD_PARTY_LICENSES.md
   python -m pip audit
   ```

综上，当前开源依赖组合可以支持优化与代理模型模块交付给客户；在随包提供第三方许可证声明、
版权声明和 NOTICE 文件后，开源许可证层面不构成侵权风险。
