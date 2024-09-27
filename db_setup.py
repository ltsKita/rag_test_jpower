import PyPDF2
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from model_download import get_embeddings

embeddings = get_embeddings()

# PDFファイルのパス
pdf_name = "第53条_まとめ資料_別添3.pdf"
pdf_path = f"data/{pdf_name}"
log_file_path = "chroma_data_log.txt"

# PDFからテキストとメタデータを抽出
def extract_text_from_pdf(pdf_path):
    texts_with_metadata = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            # ファイルごとにページ番号の開始位置、フォーマットが異なるのでここでカスタマイズ
            if text:
                # 1~3ページ目はページ番号なし、4ページ目以降は"53.8-n"形式でカスタマイズ
                if page_number <= 3:
                    metadata = {"file_name": pdf_name}  # ページ番号なし
                else:
                    custom_page_number = f"53-8-{page_number - 3}"  # ページ番号のカスタム
                    metadata = {"page": custom_page_number, "file_name": pdf_name}


                # custom_page_number = f"7.2.5-{page_number}"  # ページ番号のカスタム
                # metadata = {"page": custom_page_number, "file_name": pdf_name}
                
                texts_with_metadata.append({
                    "text": text, 
                    "metadata": metadata
                })
    return texts_with_metadata

# ログファイルへの書き込み処理
def log_to_file(log_file_path, splits_with_metadata):
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        for split in splits_with_metadata:
            log_file.write(f"Text: {split['text']}\n")
            log_file.write(f"Metadata: {split['metadata']}\n")
            log_file.write("-" * 50 + "\n")

# PDFからテキスト、ページ番号、ファイル名を抽出
texts_with_metadata = extract_text_from_pdf(pdf_path)

# テキストをチャンクに分割し、メタデータを保持
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splits_with_metadata = []

for item in texts_with_metadata:
    splits = text_splitter.split_text(item["text"])
    for split in splits:
        splits_with_metadata.append({
            "text": split,
            "metadata": item["metadata"]
        })

# Chromaデータベースにテキストとメタデータを保存
chroma_directory = "db"
vectorstore = Chroma.from_texts(
    texts=[split["text"] for split in splits_with_metadata],
    embedding=embeddings,
    metadatas=[split["metadata"] for split in splits_with_metadata],
    persist_directory=chroma_directory
)

# 保存した内容をログファイルに記録
log_to_file(log_file_path, splits_with_metadata)
