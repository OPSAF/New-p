# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 00:26:54 2025

@author: 27862
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris, load_wine
import time

# 页面设置
st.set_page_config(
    page_title="数据猜猜乐 - 数据科学小游戏",
    page_icon="🎮",
    layout="wide"
)

# 游戏标题和介绍
st.title("🎮 数据猜猜乐 - 数据科学交互游戏")
st.markdown("""
欢迎来到**数据猜猜乐**！这是一个通过互动方式学习数据科学概念的小游戏。
在下面的游戏中，你需要根据数据线索做出猜测，看看你的数据直觉如何！
""")

# 侧边栏 - 游戏选择
st.sidebar.title("游戏设置")
game_choice = st.sidebar.selectbox(
    "选择游戏模式",
    ["相关关系猜猜猜", "分类挑战赛", "聚类探索家", "异常值侦探"]
)

difficulty = st.sidebar.radio("难度级别", ["简单", "中等", "困难"])

# 缓存数据加载
@st.cache_data
def load_game_data(dataset_name):
    """加载游戏数据集"""
    if dataset_name == "iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = [data.target_names[i] for i in data.target]
    elif dataset_name == "wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    else:
        # 生成模拟数据
        np.random.seed(42)
        n_points = 100
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_points),
            'feature2': np.random.normal(0, 1, n_points),
            'cluster': np.random.choice([0, 1], n_points)
        })
    return df

# 游戏1: 相关关系猜猜猜
def correlation_game():
    st.header("🔍 相关关系猜猜猜")
    st.write("猜测两个变量之间的相关关系强度")
    
    # 生成数据
    np.random.seed(42)
    n = 50
    true_correlation = st.slider("设置真实相关系数", -1.0, 1.0, 0.7, 0.1)
    
    # 根据难度调整噪声
    noise_level = {"简单": 0.1, "中等": 0.3, "困难": 0.5}[difficulty]
    
    x = np.random.normal(0, 1, n)
    y = true_correlation * x + noise_level * np.random.normal(0, 1, n)
    
    df = pd.DataFrame({'X变量': x, 'Y变量': y})
    
    # 显示散点图但隐藏真实相关系数
    fig = px.scatter(df, x='X变量', y='Y变量', 
                    title="X和Y变量的散点图 - 猜测相关关系强度")
    st.plotly_chart(fig, use_container_width=True)
    
    # 玩家猜测
    st.subheader("你的猜测")
    guess = st.slider("你认为X和Y的相关系数大约是：", -1.0, 1.0, 0.0, 0.01)
    
    # 计算真实相关系数
    true_corr = np.corrcoef(x, y)[0, 1]
    
    if st.button("提交答案", key="corr_submit"):
        error = abs(guess - true_corr)
        score = max(0, 100 - error * 200)
        
        st.success(f"""
        **结果公布！**
        - 你的猜测: {guess:.2f}
        - 真实相关系数: {true_corr:.2f}
        - 误差: {error:.2f}
        - **得分: {score:.0f}/100**
        """)
        
        # 显示解释
        if error < 0.1:
            st.balloons()
            st.write("🎉 太棒了！你的数据直觉非常准确！")
        elif error < 0.3:
            st.write("👍 不错！你的猜测相当接近真实值。")
        else:
            st.write("💡 没关系！多练习会提高你的相关关系直觉。")

# 游戏2: 分类挑战赛
def classification_game():
    st.header("🎯 分类挑战赛")
    st.write("根据特征数据猜测样本的分类")
    
    # 加载数据
    df = load_game_data("iris")
    
    # 随机选择一个样本
    sample_idx = np.random.randint(0, len(df))
    sample = df.iloc[sample_idx]
    
    st.subheader("样本特征")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("花萼长度", f"{sample[0]:.1f} cm")
    with col2:
        st.metric("花萼宽度", f"{sample[1]:.1f} cm")
    with col3:
        st.metric("花瓣长度", f"{sample[2]:.1f} cm")
    with col4:
        st.metric("花瓣宽度", f"{sample[3]:.1f} cm")
    
    # 玩家猜测
    st.subheader("分类猜测")
    options = ['setosa', 'versicolor', 'virginica']
    guess = st.radio("你认为这个样本属于哪一类鸢尾花？", options)
    
    # 显示所有样本的分布（给提示）
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], 
                    color='target_name', title="所有样本分布图（参考）")
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("提交分类答案", key="class_submit"):
        true_class = sample['target_name']
        is_correct = (guess == true_class)
        
        if is_correct:
            st.success(f"✅ 正确！这确实是 **{true_class}** 类鸢尾花！")
            st.balloons()
        else:
            st.error(f"❌ 不正确。这实际上是 **{true_class}** 类鸢尾花。")
        
        # 学习点
        with st.expander("📚 学习这个分类"):
            st.write(f"""
            **特征分析:**
            - 花萼长度: {sample[0]:.1f} cm
            - 花萼宽度: {sample[1]:.1f} cm  
            - 花瓣长度: {sample[2]:.1f} cm
            - 花瓣宽度: {sample[3]:.1f} cm
            
            **{true_class}** 类的典型特征:
            {get_iris_characteristics(true_class)}
            """)

def get_iris_characteristics(species):
    """返回鸢尾花种类的特征描述"""
    characteristics = {
        'setosa': '花萼较大，花瓣较小且宽，通常比较容易识别',
        'versicolor': '特征介于setosa和virginica之间，中等大小',
        'virginica': '花瓣较大且长，是三个种类中最大的'
    }
    return characteristics.get(species, "暂无描述")

# 游戏3: 聚类探索家
def clustering_game():
    st.header("🔮 聚类探索家")
    st.write("猜测数据中隐藏的聚类模式")
    
    # 生成聚类数据
    np.random.seed(42)
    n_points = 100
    n_clusters = st.slider("数据中的真实聚类数", 2, 5, 3)
    
    # 生成聚类数据
    X, y_true = make_blobs(n_samples=n_points, centers=n_clusters, 
                          cluster_std=0.8, random_state=42)
    
    df = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'true_cluster': y_true})
    
    # 显示数据（隐藏真实标签）
    fig = px.scatter(df, x='X', y='Y', title="数据分布 - 猜测有多少个聚类")
    st.plotly_chart(fig, use_container_width=True)
    
    # 玩家猜测
    guess_n_clusters = st.slider("你认为数据中有多少个自然聚类？", 2, 5, 2)
    
    if st.button("查看聚类结果", key="cluster_submit"):
        # 显示真实聚类
        fig_true = px.scatter(df, x='X', y='Y', color='true_cluster',
                            title=f"真实聚类结构（{n_clusters}个聚类）")
        st.plotly_chart(fig_true, use_container_width=True)
        
        # 计算得分
        score = 100 if guess_n_clusters == n_clusters else 0
        
        if score == 100:
            st.success(f"✅ 正确！数据中确实有 {n_clusters} 个自然聚类。")
            st.balloons()
        else:
            st.error(f"❌ 不正确。数据中实际上有 {n_clusters} 个自然聚类。")
        
        st.write(f"**得分: {score}/100**")

def make_blobs(n_samples, centers, cluster_std, random_state):
    """简化版的make_blobs函数"""
    np.random.seed(random_state)
    n_features = 2
    X = []
    y = []
    
    for i in range(centers):
        center = np.random.uniform(-5, 5, n_features)
        cluster_points = np.random.normal(center, cluster_std, (n_samples//centers, n_features))
        X.extend(cluster_points)
        y.extend([i] * (n_samples//centers))
    
    return np.array(X), np.array(y)

# 游戏4: 异常值侦探
def outlier_detection_game():
    st.header("🕵️ 异常值侦探")
    st.write("找出数据中的异常值点")
    
    # 生成包含异常值的数据
    np.random.seed(42)
    n_normal = 95
    n_outliers = 5
    
    # 正常数据
    x_normal = np.random.normal(0, 1, n_normal)
    y_normal = 0.5 * x_normal + 0.3 * np.random.normal(0, 1, n_normal)
    
    # 异常值
    x_outliers = np.random.uniform(-3, 3, n_outliers)
    y_outliers = np.random.uniform(-3, 3, n_outliers)
    
    df = pd.DataFrame({
        'X': np.concatenate([x_normal, x_outliers]),
        'Y': np.concatenate([y_normal, y_outliers]),
        'is_outlier': [0]*n_normal + [1]*n_outliers
    })
    
    # 显示数据（隐藏异常值标签）
    fig = px.scatter(df, x='X', y='Y', title="数据分布 - 找出异常值点")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("异常值标记")
    st.write("使用矩形选择工具标记你认为的异常值点")
    
    # 简单版本 - 让用户输入数量
    n_guess = st.slider("你认为图中有多少个异常值点？", 0, 10, 0)
    
    if st.button("检查异常值检测", key="outlier_submit"):
        correct_outliers = len(df[df['is_outlier'] == 1])
        error = abs(n_guess - correct_outliers)
        score = max(0, 100 - error * 20)
        
        st.success(f"""
        **检测结果:**
        - 你的猜测: {n_guess} 个异常值
        - 真实数量: {correct_outliers} 个异常值
        - **得分: {score}/100**
        """)
        
        # 显示异常值
        fig_true = px.scatter(df, x='X', y='Y', color='is_outlier',
                            title="异常值检测结果（红色为异常值）")
        st.plotly_chart(fig_true, use_container_width=True)

# 主游戏路由
if game_choice == "相关关系猜猜猜":
    correlation_game()
elif game_choice == "分类挑战赛":
    classification_game()
elif game_choice == "聚类探索家":
    clustering_game()
elif game_choice == "异常值侦探":
    outlier_detection_game()

# 侧边栏 - 游戏统计
st.sidebar.markdown("---")
st.sidebar.subheader("游戏统计")
if st.sidebar.button("开始新游戏"):
    st.experimental_rerun()

st.sidebar.markdown("""
**游戏特色:**
- 互动式数据探索
- 即时反馈和学习
- 渐进式难度系统
- 实际数据集应用
""")

# 页脚
st.markdown("---")
st.markdown("### 💡 数据科学小贴士")
tips = [
    "相关关系不等于因果关系 - 总是要多思考一步",
    "数据可视化是理解数据模式的最有力工具之一",
    "异常值可能是噪音，也可能是重要信号的载体",
    "聚类分析可以帮助发现数据中隐藏的自然分组"
]

st.write(tips[np.random.randint(0, len(tips))])