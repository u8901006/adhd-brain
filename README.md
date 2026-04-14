# ADHD Brain

ADHD 文獻日報 - 每日自動從 PubMed 抓取最新 ADHD 相關文獻，由 Zhipu AI GLM-5.1 分析整理，自動生成 HTML 報告並部署到 GitHub Pages。

## 架構

- **PubMed API** - 抓取 ADHD 相關期刊的最新論文
- **Zhipu AI GLM-5.1** - 分析、摘要、分類論文
- **GitHub Actions** - 每日台北時間 11:00 自動執行
- **GitHub Pages** - 部署靜態 HTML 報告

## 搜尋範圍

關鍵字與期刊來源定義於 `ADHD_journals_keywords_pubmed_templates.md`，涵蓋：
- ADHD 專屬期刊（Journal of Attention Disorders 等）
- 兒童青少年精神醫學期刊
- 神經科學 / 神經影像期刊
- 精神藥理學期刊

## 網站

- 📊 [ADHD Brain 日報](https://u8901006.github.io/adhd-brain/)
- 🏥 [李政洋身心診所](https://www.leepsyclinic.com/)
- 📬 [訂閱電子報](https://blog.leepsyclinic.com/)

## 本地開展

```bash
pip install -r scripts/requirements.txt
python scripts/fetch_papers.py --days 7 --max-papers 40 --json --output papers.json
python scripts/generate_report.py --input papers.json --output docs/adhd-test.html --api-key $ZHIPU_API_KEY
```
