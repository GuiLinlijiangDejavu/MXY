# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import jieba
import streamlit as st
import os

class ContentBasedNewsRecommender:
    def __init__(self, data_paths=None, stopwords_path=None):
        self.data = None
        self.tfidf_vectorizer = None
        self.news_vectors = None
        self.data_paths = data_paths
        self.stopwords = self._load_stopwords(stopwords_path)

    def _load_stopwords(self, stopwords_path=None):
        """加载停用词表"""
        default_stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        return default_stopwords

    def load_data(self, data_paths=None):
        data_paths = data_paths or self.data_paths
        if not data_paths:
            raise ValueError("请提供数据路径")

        all_data = []
        for data_path in data_paths:
            print(f"加载数据: {data_path}")
            data = pd.read_csv(data_path, encoding='utf-8')
            all_data.append(data)

        # 合并所有数据
        self.data = pd.concat(all_data, ignore_index=True)

        # 数据清洗：确保标题、摘要、关键词和链接不为空
        print("数据清洗中...")
        required_columns = ['标题', '摘要', '关键词', '链接']
        self.data = self.data.dropna(subset=required_columns)

        # 文本预处理：合并标题、摘要、关键词和正文
        print("文本预处理中...")
        self.data['combined_features'] = self.data['标题'] + " " + self.data['摘要'] + " " + self.data['关键词'] + " " + self.data['正文内容']
        self.data['clean_features'] = self.data['combined_features'].apply(self._clean_text).apply(self._remove_stopwords).apply(self._segment_text)

        print(f"数据加载完成，共 {len(self.data)} 条新闻")
        return self

    def train(self):
        if self.data is None:
            raise ValueError("请先加载数据")

        print("训练TF-IDF模型...")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))  # 增加特征维度和n-gram
        self.news_vectors = self.tfidf_vectorizer.fit_transform(self.data['clean_features'])

        print("模型训练完成")
        return self

    def recommend(self, user_input, top_n=5):
        if self.data is None or self.tfidf_vectorizer is None:
            raise ValueError("模型未初始化，请先加载数据并训练")

        # 处理用户输入
        clean_input = self._segment_text(self._remove_stopwords(self._clean_text(user_input)))
        user_vector = self.tfidf_vectorizer.transform([clean_input])

        # 计算相似度
        similarities = cosine_similarity(user_vector, self.news_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]

        # 构建推荐结果
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'news_id': idx,
                'title': self.data.iloc[idx]['标题'],
                'abstract': self.data.iloc[idx]['摘要'],
                'keywords': self.data.iloc[idx]['关键词'],
                'url': self.data.iloc[idx]['链接'],
                'similarity': float(similarities[idx]),
                'publish_time': str(self.data.iloc[idx]['发布时间']) if '发布时间' in self.data.columns else '未知'
            })

        return recommendations

    def _clean_text(self, text):
        """清洗文本：去除特殊字符、数字和英文"""
        # 保留中文、基本标点符号和空格
        pattern = re.compile(r'[^ \u4e00-\u9fa5，。、；：！？（）【】《》「」]')
        return re.sub(pattern, '', text).strip()

    def _remove_stopwords(self, text):
        """去除停用词"""
        if not text:
            return ''

        # 分词后过滤停用词
        words = jieba.lcut(text)
        filtered_words = [word for word in words if word not in self.stopwords]
        return ''.join(filtered_words)

    def _segment_text(self, text):
        """中文分词"""
        return ' '.join(jieba.cut(text, cut_all=False))

def main():
    st.set_page_config(page_title="多特征新闻推荐系统", page_icon="📰", layout="wide")
    st.title("📰 新浪新闻多特征新闻推荐系统")
    st.image("新浪新闻.jpg", use_container_width=True, caption="新闻推荐系统封面")
    st.markdown("基于标题、摘要、关键词和正文的智能推荐引擎")

    with st.sidebar:
        top_n = st.slider("推荐数量", 1, 30, 10)
        st.info("""
        本系统结合新闻标题、摘要、关键词和正文进行推荐：
        - 标题：精准定位主题
        - 摘要：提炼核心内容
        - 关键词：聚焦关键信息
        - 正文： 容纳所有要素
        """)

    col1, col2 = st.columns([3, 1])

    with col1:
        user_input = st.text_area("请输入感兴趣的内容",
                                  placeholder="例如：人工智能，巴以冲突",
                                  height=100)

        if st.button("获取推荐"):
            if not user_input.strip():
                st.warning("请输入内容")
                return

            try:
                # 初始化并缓存模型
                if 'recommender' not in st.session_state:
                    st.session_state.recommender = ContentBasedNewsRecommender(["T1.csv", "T2.csv"], "stopwords.txt")
                    st.session_state.recommender.load_data().train()

                # 获取推荐结果
                recommendations = st.session_state.recommender.recommend(user_input, top_n=top_n)

                if recommendations:
                    st.subheader(f"为你推荐 {len(recommendations)} 条相关新闻：")

                    for i, news in enumerate(recommendations, 1):
                        with st.container():
                            st.markdown(f"### {i}. {news['title']}")

                            # 展示多特征信息
                            st.markdown(f"**关键词**：{news['keywords']}")
                            st.markdown(f"**摘要**：{news['abstract']}")
                            st.markdown(f"**相似度**：{news['similarity']:.2%}")
                            st.markdown(f"**发布时间**：{news['publish_time']}")
                            st.markdown(f"[查看原文]({news['url']})", unsafe_allow_html=True)
                            st.markdown("---")
                else:
                    st.info("未找到相关新闻，请调整输入")
            except Exception as e:
                st.error(f"错误：{str(e)}")

    with col2:
        st.header("使用提示")
        st.markdown("- 目前共摘录新浪新闻财经、国内、国际、体彩、娱乐五个频道约一万两千条新闻")
        st.markdown("- 输入越详细，推荐越精准")
        st.markdown("- 支持中文关键词、句子或段落")
        st.markdown("- 系统会同时匹配标题、摘要、关键词和正文")

if __name__ == "__main__":
    main()