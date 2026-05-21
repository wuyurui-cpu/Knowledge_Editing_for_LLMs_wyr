# Knowledge Editing for LLMs

## 1. 本项目简介(方向-03：大模型知识编辑)：
- 主流算法实践：基于 EasyEdit 框架，使用经典知识编辑算法。先后完成ROME 单条精准知识编辑、MEMIT 大规模批量知识编辑任务。
- 评估体系掌握：使用编辑领域三大核心评价指标，包括编辑成功率（Efficacy）、泛化性（Generalization）、局部性 / 特异性（Locality），能够从编辑效果、上下文泛化、无关知识无干扰三个维度，量化分析知识编辑算法的综合性能。
- 完整实验验证：完成单条编辑测试、500 条数据集批量注入实验，同时记录编辑过程耗时与显存占用，形成理论学习 — 算法复现 — 实验测试 — 指标评估的完整项目实践流程。

## 2. 项目结构
- `hparams`：存放编辑算法需要的超参数配置。
- `output`：存放代码运行输出的实验结果
- `baseline.py`：测试基线，在不加任何知识编辑前的模型推理测试。
- `edit_rome.py` / `edit_memit.py`：核心运行脚本，分别使用 ROME 和 MEMIT 算法对Qwen2.5-0.5B进行知识编辑。
- `evaluate.py`：评估脚本，对编辑后的模型进行打分和性能评估。
- `requirements.txt`：项目运行环境的安装与依赖。
- `datasets`：存放项目运行所需的数据集
- `Experimental Report`：存放**实验报告**

## 3. Requirements
1. Hardware Requirements
    - GPU: single NVIDIA 3090 GPU

2. Software Requirements
    - Python: 3.10
    - CUDA: 13.1 

   To install other requirements:

   ```bash
   pip install -r requirements.txt
   ```

## 4. 运行步骤
1. 基线测试
在不进行任何编辑操作的情况下，测试模型对特定过时知识或错误事实的回答:
  ```bash
  python baseline.py
  ```

2. 使用ROME算法\MEMIT算法对模型进行知识编辑：
  ```bash
  python edit_rome.py
  ```

  ```bash
  python edit_memit.py
  ```

3. 计算模型编辑的成功率\泛化性\局部性：

  ```bash
  python evaluate.py
  ```
## 5. 结果输出
实验结果将会输出到窗口以及记录到output文件夹下
