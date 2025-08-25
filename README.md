# 🟢 One4All: Modular Expert System with SLMs

**One4All** is a modular AI system that combines multiple **Specialized Language Models (SLMs)** trained on specific domains using **QLoRA**. Instead of relying on a single large LLM, One4All keeps a lightweight base LLM always loaded and dynamically loads LoRA weights for domain-specific experts. This design enables high performance with minimal hardware requirements.

---

## 🚀 How It Works (English)

1. **Question Routing**  
   When a question is asked, it first goes to the **Router**, a lightweight classifier model. The router analyzes the query and determines which domain-specific expert(s) should handle it.

2. **Expert Processing**  
   Experts are **SLMs trained on specialized datasets** (e.g., medical, technology, e-commerce). Once the router identifies the relevant domain, the question is forwarded to the corresponding expert. Each expert loads only its LoRA weights on the base model, minimizing memory usage.

3. **Why SLMs?**  
   - SLMs allow running domain-specific tasks efficiently on smaller hardware (e.g., a single GPU with 8–12GB VRAM).  
   - They reduce training and inference costs compared to a large monolithic LLM.  
   - Each expert can be independently updated without retraining the base model.

4. **Memory Efficiency**  
   Example:  
   - A base model (e.g., ParsBERT) with QLoRA LoRA weights for 3 experts may require **~12–16GB RAM per expert** loaded sequentially.  
   - A comparable LLM (7B+ parameters) may require **>40GB RAM** to run a single instance.  
   Using SLMs + LoRA reduces hardware requirements and enables modular scaling.

5. **Example Flow**
```text
Question: "Which smartphone has the best camera in 2025?"
  ↓
Router: Classifies as "Mobile/Technology"
  ↓
Expert: Mobile Expert (SLM trained on mobile reviews)
  ↓
Answer returned from expert
```

---

## 📊 Recommended SLMs for One4All

| Model Name           | Languages Supported | Domain Specialization       | QLoRA Memory Req |
|---------------------|------------------|----------------------------|----------------|
| ParsBERT            | Persian           | General / NLP tasks        | 4–6 GB         |
| mBERT               | Multi-language    | General                    | 6–8 GB         |
| CamemBERT            | French            | General                    | 4–6 GB         |
| Qwen3-SLM           | English, Multi    | Causal Language Modeling   | 8–12 GB        |
| SaadiBERT            | Persian           | Medical / Healthcare       | 4–6 GB         |

> One4All can dynamically load multiple SLM experts depending on the task.

---

## 🔧 Technical Notes

- **Base model** always loaded in memory.  
- **LoRA adapters** applied for each expert separately.  
- Router is a **lightweight classification model** that routes queries efficiently.  
- Supports **multi-domain modular updates** without retraining the base LLM.  

**References:**  
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021  
- Qwen3 model papers and SLM studies in Persian and English NLP  



✅ **Status:** پروژه در حال آپلود و انجام تست‌های نهایی است.
