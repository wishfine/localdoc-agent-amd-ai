# LocalDoc Agent — 系统设计文档

> 面向 AMD 锐龙 AI MAX+ 平台的本地知识库智能体系统设计

---

## 目录

1. [整体架构设计](#1-整体架构设计)
2. [各模块设计说明](#2-各模块设计说明)
3. [异构调度策略设计](#3-异构调度策略设计)
4. [后端抽象层设计](#4-后端抽象层设计)
5. [数据流图](#5-数据流图)
6. [关键算法说明](#6-关键算法说明)

---

## 1. 整体架构设计

### 1.1 设计目标

本系统的核心设计目标是：

1. **本地化**：所有数据处理与推理均在本地完成，保护文档隐私。
2. **异构优化**：充分利用 AMD Ryzen AI MAX+ 的 CPU + iGPU + NPU 三类计算单元。
3. **可扩展性**：模块化设计，各组件可独立替换与升级。
4. **健壮性**：在无 GPU/NPU 硬件时，自动回退到 CPU 模拟模式，确保系统可用。

### 1.2 架构分层

系统采用四层架构设计：

```
┌─────────────────────────────────────────────┐
│           交互层 (Interaction Layer)          │
│    Gradio Web UI / CLI / API 接口            │
├─────────────────────────────────────────────┤
│           编排层 (Orchestration Layer)        │
│    Agent Pipeline — 任务编排与状态管理        │
├─────────────────────────────────────────────┤
│           计算层 (Computation Layer)          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │  CPU 后端 │  │  GPU 后端 │  │  NPU 后端│    │
│  └─────────┘  └─────────┘  └─────────┘     │
│    调度器：任务分类 → 后端选择 → 结果聚合     │
├─────────────────────────────────────────────┤
│           数据层 (Data Layer)                 │
│    文档存储 / 向量索引 / 缓存 / 配置          │
└─────────────────────────────────────────────┘
```

### 1.3 核心 Pipeline

文档到问答的端到端流程：

```
原始文档 → [Loader] → 纯文本
    → [Chunker] → 文本块列表
    → [Embedder] → 向量集合
    → [VectorStore] → 向量索引
    → [Retriever] → 相关文本块
    → [Generator] → 最终答案
```

每个阶段都可以独立调用后端进行加速计算。

---

## 2. 各模块设计说明

### 2.1 Loader 模块 — 文档加载

**职责**：将不同格式的文档解析为纯文本字符串。

**设计要点**：

- 采用**策略模式**，定义 `BaseLoader` 抽象基类，各格式实现具体加载器。
- 支持格式：PDF、TXT、Markdown。
- 统一输出格式：`List[Document]`，其中 `Document` 包含 `content`（文本内容）和 `metadata`（来源、页码等元数据）。

**类结构**：

```python
class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> List[Document]: ...

class PDFLoader(BaseLoader):
    def load(self, file_path: str) -> List[Document]:
        # 使用 PyPDF2 或 pdfplumber 提取文本
        ...

class TXTLoader(BaseLoader):
    def load(self, file_path: str) -> List[Document]:
        # 直接读取 UTF-8 文本
        ...

class MarkdownLoader(BaseLoader):
    def load(self, file_path: str) -> List[Document]:
        # 读取 Markdown，可选去除格式标记
        ...
```

**调度策略**：Loader 为 I/O 密集型任务，固定在 **CPU** 上执行。

---

### 2.2 Chunker 模块 — 文本切块

**职责**：将长文本按策略切分为适合嵌入的短文本块。

**设计要点**：

- 支持两种切块策略：
  - **滑动窗口切块**（Sliding Window）：固定窗口大小 + 重叠区域。
  - **语义边界切块**（Semantic Chunking）：按段落/句子边界切块，保持语义完整性。
- 切块参数可配置：`chunk_size`（块大小）、`chunk_overlap`（重叠大小）。
- 每个块保留上下文信息，避免语义断裂。

**滑动窗口算法**：

```
输入: text, chunk_size=512, overlap=64
输出: List[Chunk]

pos = 0
chunks = []
while pos < len(text):
    end = min(pos + chunk_size, len(text))
    chunk = text[pos:end]
    chunks.append(Chunk(content=chunk, start=pos, end=end))
    pos += chunk_size - overlap  # 步进 = 窗口大小 - 重叠
return chunks
```

**调度策略**：Chunker 为字符串处理任务，固定在 **CPU** 上执行。

---

### 2.3 Embedding 模块 — 向量嵌入

**职责**：将文本块转换为数值向量，用于后续的相似度计算。

**设计要点**：

- **TF-IDF 嵌入**（默认）：使用词频-逆文档频率统计方法，计算简单高效，CPU 即可高效运行。
- **密集向量嵌入**（扩展接口）：预留深度学习模型嵌入接口，可对接 NPU 加速的 Transformer 模型。
- 向量存储为 NumPy 数组，支持批量操作。

**TF-IDF 计算**：

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

其中:
  TF(t, d) = 词 t 在文档 d 中的出现次数 / 文档 d 的总词数
  IDF(t) = log(总文档数 / 包含词 t 的文档数)
```

**调度策略**：
- TF-IDF 计算：**CPU**（稀疏矩阵运算，CPU 更高效）
- 密集向量嵌入：**GPU** 或 **NPU**（大规模矩阵乘法）

---

### 2.4 Retriever 模块 — 语义检索

**职责**：根据用户查询，从向量索引中检索最相关的文本块。

**设计要点**：

- 核心算法：**余弦相似度** Top-K 检索。
- 支持批量查询优化。
- 返回结果包含：文本块内容、相似度得分、元数据。

**检索算法**：

```
输入: query_vector, document_vectors, top_k=5
输出: List[(chunk, score)]

# 计算查询向量与所有文档向量的余弦相似度
scores = cosine_similarity(query_vector, document_vectors)

# 取 Top-K
top_indices = argsort(scores, descending=True)[:top_k]

return [(chunks[i], scores[i]) for i in top_indices]
```

**调度策略**：
- 小规模检索（< 1000 向量）：**CPU**
- 大规模检索（>= 1000 向量）：**GPU**（批量向量点积并行化）

---

### 2.5 Generator 模块 — 答案生成

**职责**：根据检索到的相关文本块，生成最终回答。

**设计要点**：

- **模板抽取式生成**（默认）：将检索到的文本块拼接，按模板格式化输出。
- **LLM 生成接口**（扩展）：预留大语言模型接入接口，可对接本地量化模型。
- 生成策略可根据查询类型动态调整。

**模板生成逻辑**：

```python
def generate(self, query: str, contexts: List[Chunk]) -> str:
    # 按相关度排序
    sorted_contexts = sorted(contexts, key=lambda c: c.score, reverse=True)

    # 构建答案
    answer_parts = []
    for i, ctx in enumerate(sorted_contexts[:3]):
        answer_parts.append(f"【参考{i+1}】{ctx.content[:200]}")

    return f"关于「{query}」的回答：\n\n" + "\n\n".join(answer_parts)
```

**调度策略**：生成任务为逻辑密集型，固定在 **CPU** 上执行。

---

## 3. 异构调度策略设计

### 3.1 调度器架构

异构调度器（`HeterogeneousScheduler`）是系统的核心组件之一，
负责将计算任务分配到最合适的硬件后端。

```
                 ┌─────────────────────────────┐
                 │      HeterogeneousScheduler  │
                 ├─────────────────────────────┤
                 │  1. 任务特征分析              │
                 │     - 计算类型（I/O/逻辑/矩阵）│
                 │     - 数据规模                 │
                 │     - 并行度需求               │
                 ├─────────────────────────────┤
                 │  2. 后端选择                   │
                 │     - 后端可用性检测           │
                 │     - 后端负载评估             │
                 │     - 任务-后端匹配表查询      │
                 ├─────────────────────────────┤
                 │  3. 任务分发与执行             │
                 │     - 同步/异步执行            │
                 │     - 超时控制                 │
                 │     - 错误处理与回退           │
                 ├─────────────────────────────┤
                 │  4. 结果聚合                   │
                 │     - 结果格式统一             │
                 │     - 性能指标记录             │
                 └─────────────────────────────┘
```

### 3.2 任务分类规则

| 任务类型 | 特征描述 | 匹配后端 |
|----------|----------|----------|
| `IO_TASK` | 文件读写、网络请求 | CPU |
| `LOGIC_TASK` | 字符串处理、规则匹配 | CPU |
| `SPARSE_TASK` | 稀疏矩阵运算 | CPU |
| `DENSE_TASK` | 密集矩阵乘法（GEMM） | GPU > NPU > CPU |
| `INFERENCE_TASK` | 神经网络前向推理 | NPU > GPU > CPU |
| `BATCH_TASK` | 大规模并行计算 | GPU > CPU |

### 3.3 后端选择算法

```python
def select_backend(self, task_type: TaskType, data_size: int) -> Backend:
    # 1. 查询匹配表，获取候选后端列表（按优先级排序）
    candidates = self._match_table[task_type]

    # 2. 过滤不可用的后端
    available = [b for b in candidates if b.is_available()]

    # 3. 对于小数据量任务，直接使用 CPU（避免传输开销）
    if data_size < self._transfer_threshold:
        return self._cpu_backend

    # 4. 返回最高优先级的可用后端
    if available:
        return available[0]

    # 5. 回退到 CPU
    return self._cpu_backend
```

### 3.4 回退机制

当目标后端不可用时，系统按以下顺序回退：

```
NPU → GPU → CPU
```

每次回退都会记录日志，并在性能报告中标注实际使用的后端。

---

## 4. 后端抽象层设计

### 4.1 统一接口定义

所有后端实现统一的 `BaseBackend` 接口：

```python
class BaseBackend(ABC):
    """计算后端抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称标识"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检测后端是否可用"""
        ...

    @abstractmethod
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """矩阵乘法"""
        ...

    @abstractmethod
    def cosine_similarity(self, query: np.ndarray,
                          documents: np.ndarray) -> np.ndarray:
        """余弦相似度计算"""
        ...

    @abstractmethod
    def dot_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """向量点积"""
        ...

    def get_info(self) -> dict:
        """获取后端信息（设备名称、内存等）"""
        return {"name": self.name, "available": self.is_available()}
```

### 4.2 后端实现

| 后端 | 类名 | 实现方式 | 适用场景 |
|------|------|----------|----------|
| CPU | `CPUBackend` | NumPy | 所有任务（基准回退） |
| GPU | `AMDGPUBackend` | ROCm/HIP 或 CuPy | 大规模矩阵运算 |
| NPU | `AMDNPUBackend` | ONNX Runtime (XDNA EP) | 神经网络推理 |
| 模拟NPU | `SimulatedNPUBackend` | NumPy 模拟 | 无 NPU 硬件时的开发测试 |

### 4.3 后端注册机制

```python
# localdoc/backends/__init__.py
from .cpu_backend import CPUBackend
from .gpu_backend import AMDGPUBackend
from .npu_backend import AMDNPUBackend
from .simulated_npu import SimulatedNPUBackend

# 后端注册表：按优先级排序
BACKEND_REGISTRY = [
    ("npu", AMDNPUBackend),
    ("gpu", AMDGPUBackend),
    ("cpu", CPUBackend),
]

# 自动检测并实例化可用后端
def get_available_backends():
    backends = []
    for name, cls in BACKEND_REGISTRY:
        instance = cls()
        if instance.is_available():
            backends.append(instance)
    # 始终包含 CPU 后端作为兜底
    if not any(isinstance(b, CPUBackend) for b in backends):
        backends.append(CPUBackend())
    return backends
```

---

## 5. 数据流图

### 5.1 端到端数据流

```
用户上传文档
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ [1] Loader: 文档解析                                     │
│  输入: file_path (str)                                   │
│  输出: documents (List[Document])                        │
│  后端: CPU                                               │
│  数据格式: {"content": str, "metadata": dict}            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ [2] Chunker: 文本切块                                    │
│  输入: documents (List[Document])                        │
│  输出: chunks (List[Chunk])                              │
│  后端: CPU                                               │
│  数据格式: {"content": str, "index": int, "source": str} │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ [3] Embedder: 向量嵌入                                   │
│  输入: chunks (List[Chunk])                              │
│  输出: embeddings (np.ndarray, shape=[N, D])             │
│  后端: CPU (TF-IDF) / GPU/NPU (Dense)                   │
│  数据格式: float32 数组                                   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ [4] VectorStore: 向量索引存储                             │
│  输入: embeddings + chunks                               │
│  输出: index (内存中的向量索引)                            │
│  后端: CPU (内存管理)                                     │
└───────────────────────┬─────────────────────────────────┘
                        │
    用户提出问题 ─────────┤
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ [5] Retriever: 语义检索                                  │
│  输入: query_vector, index, top_k                        │
│  输出: results (List[(Chunk, score)])                    │
│  后端: CPU (小规模) / GPU (大规模)                        │
│  算法: 余弦相似度 Top-K                                   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ [6] Generator: 答案生成                                  │
│  输入: query (str), contexts (List[Chunk])               │
│  输出: answer (str)                                      │
│  后端: CPU (模板) / NPU (LLM 推理)                       │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
              返回答案给用户
```

### 5.2 异构计算数据流

```
                   ┌──────────────┐
                   │   任务队列    │
                   └──────┬───────┘
                          │
                   ┌──────▼───────┐
                   │   调度器      │
                   └──────┬───────┘
                          │
           ┌──────────────┼──────────────┐
           │              │              │
    ┌──────▼──────┐ ┌─────▼─────┐ ┌─────▼─────┐
    │   CPU 后端   │ │  GPU 后端  │ │  NPU 后端  │
    │  ┌────────┐ │ │ ┌───────┐ │ │ ┌───────┐ │
    │  │Loader  │ │ │ │Dense  │ │ │ │ONNX   │ │
    │  │Chunker │ │ │ │Matrix │ │ │ │Infere-│ │
    │  │TF-IDF  │ │ │ │Compute│ │ │ │nce    │ │
    │  │Template│ │ │ │Batch  │ │ │ │       │ │
    │  └────────┘ │ │ └───────┘ │ │ └───────┘ │
    └──────┬──────┘ └─────┬─────┘ └─────┬─────┘
           │              │              │
           └──────────────┼──────────────┘
                          │
                   ┌──────▼───────┐
                   │   结果聚合    │
                   └──────────────┘
```

---

## 6. 关键算法说明

### 6.1 文本切块策略

#### 滑动窗口切块（Sliding Window Chunking）

**算法描述**：

滑动窗口切块是最基础的切块方法，以固定大小的窗口在文本上滑动，
窗口之间保留一定的重叠区域以保证上下文连续性。

**算法步骤**：

```
输入:
  text       - 原始文本字符串
  chunk_size - 每个块的目标大小（字符数），默认 512
  overlap    - 相邻块的重叠区域大小（字符数），默认 64

输出:
  chunks     - 切块结果列表

过程:
  1. 初始化: pos = 0, chunks = []
  2. 循环直到 pos >= len(text):
     a. 计算当前块的起止位置:
        start = pos
        end = min(pos + chunk_size, len(text))
     b. 提取文本块: chunk = text[start:end]
     c. 记录元信息: chunks.append({
            "content": chunk,
            "start": start,
            "end": end,
            "index": len(chunks)
        })
     d. 窗口滑动: pos += chunk_size - overlap
  3. 返回 chunks
```

**复杂度分析**：
- 时间复杂度: O(N)，其中 N 为文本长度
- 空间复杂度: O(N)，需要存储所有切块结果

**参数调优建议**：
- `chunk_size` 过大：语义过于宽泛，检索精度下降
- `chunk_size` 过小：上下文不足，语义断裂
- `overlap` 过小：相邻块之间缺乏关联
- `overlap` 过大：冗余数据增加，存储与计算开销上升
- 推荐: `chunk_size = 512`, `overlap = 64`（约为 chunk_size 的 12.5%）

#### 语义边界切块（Semantic Chunking）

**算法描述**：

语义边界切块在滑动窗口的基础上，优先在自然语义边界处切分，
避免在句子或段落中间断开。

**算法步骤**：

```
输入:
  text         - 原始文本
  max_size     - 最大块大小
  separators   - 优先级递减的分隔符列表 ["\n\n", "\n", "。", ".", " "]

输出:
  chunks       - 语义完整的文本块列表

过程:
  1. 使用最高优先级的分隔符将文本分割为片段
  2. 遍历片段，合并相邻片段直到达到 max_size
  3. 如果单个片段超过 max_size，降级使用下一级分隔符继续分割
  4. 返回所有块
```

---

### 6.2 TF-IDF 嵌入算法

#### 算法原理

TF-IDF（Term Frequency-Inverse Document Frequency）是一种经典的文本特征提取方法，
通过统计词频和逆文档频率来衡量词语对文档的重要程度。

#### 计算公式

```
给定文档集合 D = {d₁, d₂, ..., dₙ}，词项 t，文档 d:

1. 词频 (Term Frequency):
   TF(t, d) = count(t, d) / |d|

   其中 count(t, d) 为词 t 在文档 d 中的出现次数，
   |d| 为文档 d 的总词数。

2. 逆文档频率 (Inverse Document Frequency):
   IDF(t) = log(|D| / |{d ∈ D : t ∈ d}|)

   其中 |D| 为文档总数，
   |{d ∈ D : t ∈ d}| 为包含词 t 的文档数量。

3. TF-IDF 值:
   TF-IDF(t, d) = TF(t, d) × IDF(t)
```

#### 向量化过程

```
输入: chunks (List[str]) — N 个文本块

步骤:
  1. 构建词汇表 V = {所有文本块中的唯一词项}
     - 中文需要分词（使用 jieba 或简单字符 n-gram）
     - 去除停用词（的、了、是、在、...）

  2. 计算 IDF 值（对整个文档集合）
     对每个词项 t ∈ V:
       IDF[t] = log(N / df(t))

  3. 对每个文本块计算 TF-IDF 向量
     对第 i 个文本块 dᵢ:
       vector_i[t] = TF(t, dᵢ) × IDF[t]  对每个 t ∈ V

输出: embeddings (np.ndarray, shape=[N, |V|]) — 稀疏向量矩阵
```

#### 后端调度

TF-IDF 计算主要涉及词频统计和对数运算，属于稀疏计算范畴。
CPU 在稀疏矩阵运算上效率更高（避免 GPU 的内存传输开销），
因此 **TF-IDF 嵌入固定在 CPU 后端执行**。

---

### 6.3 余弦相似度检索算法

#### 算法原理

余弦相似度通过计算两个向量夹角的余弦值来衡量它们的方向相似性。
值域为 [-1, 1]，其中 1 表示完全相同方向，0 表示正交，-1 表示完全相反。

#### 计算公式

```
给定查询向量 q 和文档向量 d:

cosine_similarity(q, d) = (q · d) / (||q|| × ||d||)

其中:
  q · d = Σᵢ(qᵢ × dᵢ)          — 向量点积
  ||q|| = √(Σᵢ qᵢ²)             — q 的 L2 范数
  ||d|| = √(Σᵢ dᵢ²)             — d 的 L2 范数
```

#### 批量检索算法

```
输入:
  query       - 查询向量 (shape: [D])
  documents   - 文档向量矩阵 (shape: [N, D])
  top_k       - 返回前 K 个结果

过程:
  1. 归一化:
     query_norm = query / ||query||
     docs_norm  = documents / ||documents|| (逐行归一化)

  2. 批量余弦相似度:
     scores = docs_norm @ query_norm  (矩阵-向量乘法, shape: [N])

  3. Top-K 选择:
     top_indices = argsort(scores, descending=True)[:top_k]

  4. 构建结果:
     results = [(chunks[i], scores[i]) for i in top_indices]

输出: results — 按相似度降序排列的 Top-K 结果
```

#### 后端调度策略

| 数据规模 | 推荐后端 | 理由 |
|----------|----------|------|
| N < 100 | CPU | 数据量小，传输开销 > 计算收益 |
| 100 ≤ N < 10000 | CPU | NumPy 优化的矩阵乘法已足够快 |
| N ≥ 10000 | GPU | 大规模并行计算显著加速 |

#### 性能优化技巧

1. **向量预归一化**：在存储阶段预先对文档向量做 L2 归一化，
   检索时只需计算点积，省去除法运算。
2. **批量计算**：使用矩阵乘法一次性计算所有相似度，
   避免逐个向量循环。
3. **SIMD 优化**：CPU 后端利用 NumPy 底层的 SIMD 指令加速向量运算。

---

## 附录：配置参数说明

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `chunk_size` | 512 | 文本块大小（字符数） |
| `chunk_overlap` | 64 | 相邻块重叠大小 |
| `top_k` | 5 | 检索返回结果数 |
| `embedding_method` | "tfidf" | 嵌入方法选择 |
| `transfer_threshold` | 1000 | 启用 GPU 的最小数据量 |
| `backend_timeout` | 30 | 后端执行超时（秒） |
