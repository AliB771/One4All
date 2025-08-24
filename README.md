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

---

# 🟢 One4All: سیستم متخصص ماژولار با SLMها

**One4All** یک سیستم هوش مصنوعی ماژولار است که ترکیبی از **مدل‌های زبانی تخصصی (SLM)** را با استفاده از **QLoRA** پیاده‌سازی می‌کند. به جای استفاده از یک مدل زبانی بزرگ، One4All همیشه مدل پایه سبک را لود نگه می‌دارد و فقط وزن‌های LoRA مربوط به متخصص‌ها را بارگذاری می‌کند.

---

## 🚀 نحوه عملکرد (فارسی)

1. **مسیردهی پرسش‌ها**  
   وقتی سوالی پرسیده می‌شود، ابتدا به **Router** می‌رود. این مدل سبک، زمینه‌ی سوال را شناسایی می‌کند و مشخص می‌کند کدام متخصص یا متخصص‌ها باید به آن پاسخ دهند.

2. **پردازش توسط متخصص‌ها**  
   متخصص‌ها **SLMهای آموزش دیده در حوزه‌های خاص** هستند (مثلاً پزشکی، تکنولوژی، خرید). پس از شناسایی حوزه توسط Router، سوال به متخصص مربوطه ارسال می‌شود و تنها وزن‌های LoRA آن متخصص روی مدل پایه بارگذاری می‌شوند تا حافظه بهینه شود.

3. **چرا SLM؟**  
   - SLMها اجرای کارهای تخصصی را با سخت‌افزار کم (مثلاً یک GPU با 8–12GB VRAM) امکان‌پذیر می‌کنند.  
   - هزینه‌های آموزش و استنتاج نسبت به LLMهای بزرگ کمتر است.  
   - هر متخصص به‌صورت مستقل قابل آپدیت است بدون نیاز به آموزش مجدد مدل پایه.

4. **کارایی حافظه**  
   مثال:  
   - مدل پایه (ParsBERT) با وزن‌های LoRA سه متخصص: **~12–16GB RAM** برای هر متخصص به‌صورت متوالی کافی است.  
   - یک LLM بزرگ (7B+ پارامتر) نیازمند **>40GB RAM** برای یک نمونه است.  
   بنابراین، SLMها + LoRA نیاز سخت‌افزاری را کاهش داده و مقیاس‌پذیری را آسان می‌کنند.

5. **مثال جریان پرسش**
```text
سوال: "بهترین دوربین گوشی در 2025 کدام است؟"
  ↓
Router: دسته‌بندی "موبایل / تکنولوژی"
  ↓
Expert: متخصص موبایل (SLM آموزش دیده روی بررسی‌های موبایل)
  ↓
پاسخ از متخصص برگشت داده می‌شود
```

---

## 📊 SLMهای توصیه شده برای One4All

| نام مدل             | زبان‌های پشتیبانی شده | حوزه تخصصی                  | نیاز حافظه QLoRA |
|--------------------|--------------------|-----------------------------|----------------|
| ParsBERT            | فارسی               | عمومی / NLP                | 4–6 GB         |
| mBERT               | چندزبان             | عمومی                       | 6–8 GB         |
| CamemBERT           | فرانسه              | عمومی                       | 4–6 GB         |
| Qwen3-SLM           | انگلیسی، چندزبان    | مدل‌سازی زبان سببی (Causal) | 8–12 GB        |
| SaadiBERT           | فارسی               | پزشکی / سلامت               | 4–6 GB         |

> One4All به صورت داینامیک چند متخصص SLM را بسته به پرسش بارگذاری می‌کند.

---

## 🔧 نکات فنی

- **مدل پایه** همیشه در حافظه نگه داشته می‌شود.  
- **LoRA adapters** برای هر متخصص به‌صورت جداگانه اعمال می‌شوند.  
- Router مدل سبک و کارآمدی است که پرسش‌ها را مسیردهی می‌کند.  
- پشتیبانی از **آپدیت ماژولار چند حوزه‌ای** بدون نیاز به آموزش مجدد مدل پایه.  

**مراجع:**  
- Hu و همکاران، *LoRA: Low-Rank Adaptation of Large Language Models*, 2021  
- مقالات Qwen3 و پژوهش‌های SLM در حوزه NLP فارسی و انگلیسی  

---

✅ **Status:** پروژه در حال آپلود و انجام تست‌های نهایی است.

