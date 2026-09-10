# Hiver AI Customer Support Agent

An evidence-first AI customer-support agent built for the Hiver SDE Intern take-home assignment.

The system uses historical customer-support conversations from Kaggle's **Customer Support on Twitter** dataset to:

1. Classify an incoming customer message into a small, data-derived intent taxonomy.
2. Retrieve historically similar customer-support cases.
3. Draft a response grounded in how the brand historically handled similar issues.
4. Decide whether the message should be automatically handled or escalated to a human.
5. Provide the evidence and reasoning behind the decision.

The project focuses on **grounded support automation rather than unconstrained chatbot generation**.

---

## 1. Problem Statement

Customer-support teams receive a large number of repetitive questions every day.

For many messages, the support team has already solved similar problems in the past. However, agents still need to manually:

- Understand what the customer is asking.
- Search previous conversations.
- Determine how similar cases were resolved.
- Write an appropriate response.
- Decide whether the issue can be safely handled automatically or should be escalated.

This project explores whether historical support conversations can be used to build an AI support agent that assists with these tasks while keeping an evidence trail.

---

## 2. Selected Brand

The project uses:

**AmazonHelp**

from Kaggle's:

**Customer Support on Twitter**

dataset.

The brand was selected using a deterministic scoring rule based on the volume of outbound support activity and direct replies in the sampled dataset.

The project does not assume that historical Twitter responses represent Amazon's current official support policy. Historical conversations are used as **evidence of past resolution patterns**, not as authoritative current policy.

---

## 3. Dataset

Source:

**Customer Support on Twitter**

Kaggle dataset:

`thoughtvector/customer-support-on-twitter`

The original dataset contains millions of tweets from customer-support conversations between customers and brands.

For reproducibility and practical local development, this project uses a deterministic subset of the available CSV data rather than processing the entire dataset.

### Local dataset statistics

The preparation pipeline currently reads a deterministic prefix of:

- Rows read: `500,000`
- Chunk size: `100,000`
- Random seed: `42`
- Source file size: approximately `516 MB`

The selected brand is:

`AmazonHelp`

The resulting preparation pipeline identified:

- Customer tweets: `39,299`
- Support-agent tweets: `39,419`
- Conversations: `39,299`
- Conversations with an agent response: `39,299`
- Usable resolved examples: `39,299`

The average and median conversation length in the selected examples are both approximately:

`2 messages`

The raw dataset is intentionally **not committed to GitHub** because of its size.

---

## 4. Architecture

```text
                         ┌──────────────────────┐
                         │     React + Vite      │
                         │      Frontend        │
                         └──────────┬───────────┘
                                    │
                                    │ HTTP
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │       Backend        │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     │              │              │
                     ▼              ▼              ▼
                Intent Model    Retriever      Decision Logic
                     │              │              │
                     │              ▼              │
                     │       FAISS Index           │
                     │              │              │
                     │              ▼              │
                     │     Historical Support      │
                     │       Conversations         │
                     │                             │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                           Grounded Reply Draft
                                    │
                                    ▼
                         AUTO_HANDLE / ESCALATE
