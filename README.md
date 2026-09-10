# 🤖 Hiver AI Customer Support Agent

### Hiver SDE Intern Take-Home Assignment


## 📌 Overview

This project is an **AI-powered customer support agent** built for the Hiver SDE Intern take-home assignment.

The system learns from historical customer-support conversations to automate repetitive support tasks.

### Workflow

Customer Message  
↓  
Intent Classification  
↓  
Find Similar Historical Cases  
↓  
Use Historical Evidence  
↓  
Generate Response  
↓  
AUTO_HANDLE / ESCALATE

> The main idea is to use real historical support conversations as evidence instead of allowing the AI to generate unsupported answers.

---

## 🎯 What the Agent Does

The system:

- **Classifies** the customer's issue into a small set of intents.
- **Retrieves** similar historical support conversations.
- **Generates** a response based on historical resolution patterns.
- **Decides** whether to automatically handle the request or escalate it to a human.

---

## 🏢 Selected Brand

**AmazonHelp**

The project uses AmazonHelp conversations from Kaggle's **Customer Support on Twitter** dataset:

`thoughtvector/customer-support-on-twitter`

Historical responses are treated as examples of past support behavior, not as current official Amazon policies.

---

## 📊 Dataset

For reproducibility, a deterministic subset of the dataset is used.

| Metric | Value |
|---|---:|
| Rows processed | 500,000 |
| Chunk size | 100,000 |
| Random seed | 42 |
| Source size | ~516 MB |
| Customer tweets | 39,299 |
| Support tweets | 39,419 |
| Conversations | 39,299 |
| Resolved examples | 39,299 |

The raw dataset is not included in GitHub because of its size.

---

## 🧠 Intent Classification

| Intent | Description |
|---|---|
| `prime_membership` | Prime membership and benefits |
| `account_or_security` | Account access and security issues |
| `payment_or_pricing` | Payments, billing, pricing and refunds |
| `product_availability` | Product availability |
| `technical_or_content` | Technical and content-related issues |
| `customer_service` | General support requests |
| `other_or_unclear` | Unclear or unmatched requests |

---

## 🔎 Historical Retrieval

The system uses **Sentence Transformers + FAISS** to find similar historical customer-support cases.

Current retrieval index:

`39,099 leakage-safe historical cases`

Retrieved conversations are used as evidence when generating the response.

---

## 🚦 Escalation

The agent produces one of two decisions:

- `AUTO_HANDLE`
- `ESCALATE`

Sensitive, ambiguous, low-confidence, or insufficiently supported requests should be escalated instead of receiving an unsupported automated answer.

### Example

`"Someone has accessed my Amazon account and I think it has been compromised."`

This is treated as a security-sensitive request and should be handled conservatively.

---

## 🖥️ Technology Stack

### Frontend

- React
- Vite
- JavaScript
- CSS

### Backend

- Python
- FastAPI
- Uvicorn

### AI / Retrieval

- Sentence Transformers
- FAISS
- Pandas
- NumPy

### Evaluation

- Classification metrics
- Retrieval metrics
- LLM-as-Judge
- Human evaluation

---

## 📁 Project Structure

```text
hiver-project/
├── frontend/       # React + Vite application
├── backend/        # FastAPI backend and ML pipeline
├── data/           # Processed and evaluation data
├── evaluation/     # Evaluation and baselines
├── report/         # Assignment report
├── tests/          # Tests
└── README.md

## 🚀 Quick Start

### Backend

```powershell
cd D:\hiver-project
.venv\Scripts\Activate.ps1
cd backend
uvicorn app:app --reload --port 8001
```

Backend:

`http://127.0.0.1:8001`

API documentation:

`http://127.0.0.1:8001/docs`

### Frontend

Open another terminal:

```powershell
cd D:\hiver-project\frontend
npm install
npm run dev
```

Open the URL provided by Vite, normally:

`http://localhost:5173`

---

## 🧪 Example

### Input

```text
I can't access my Amazon account.
```

### Process

```text
Classify the issue
        ↓
Find similar AmazonHelp cases
        ↓
Use historical responses as evidence
        ↓
Generate a grounded reply
        ↓
Decide AUTO_HANDLE or ESCALATE
```

---

## 📈 Evaluation

The system is evaluated against simple baselines.

Evaluation covers:

- Intent classification
- Retrieval quality
- Response relevance
- Response groundedness
- Response helpfulness
- Escalation performance

A separate **150–250 example golden evaluation set** is used for final evaluation.

Human labels are required for the final golden set. AI-generated labels are only suggestions and are not treated as human evaluation.

The project also includes an **LLM-as-Judge** approach and compares it with human ratings where available.

---

## ⚠️ Limitations

- Historical conversations do not represent current company policy.
- Only a subset of the complete dataset is used.
- Some messages are ambiguous.
- `other_or_unclear` is currently a large category.
- Similarity does not always mean the retrieved case is correct.
- LLM-generated responses can still contain unsupported information.
- LLM-as-Judge is not a replacement for human evaluation.
- This is a prototype and does not perform real customer-account actions.

---

## 🔮 Future Improvements

With more development time, I would:

- Refine the intent taxonomy.
- Improve retrieval using intent + semantic similarity.
- Add confidence calibration.
- Strengthen escalation for sensitive cases.
- Expand human evaluation.
- Combine historical conversations with current authoritative support documentation.

---

## 💡 Key Design Principle

```text
Understand
    ↓
Classify
    ↓
Retrieve Evidence
    ↓
Generate Grounded Response
    ↓
Evaluate
    ↓
Handle or Escalate
```

The goal is not simply to build a chatbot that generates fluent responses.

The goal is to build a **reliable, evidence-grounded customer-support assistant that knows when to automate and when to involve a human.**

---

## 👩‍💻 Author

### Monigasri M

**Computer Science Engineering Student**

**Bannari Amman Institute of Technology**

**Madurai, Tamil Nadu, India**

🔗 **LinkedIn:** https://www.linkedin.com/in/monigasri-m-9281a12b0

🔗 **GitHub:** https://github.com/Monigasri

---

### Hiver SDE Intern Take-Home Assignment

Built by **Monigasri M**.
