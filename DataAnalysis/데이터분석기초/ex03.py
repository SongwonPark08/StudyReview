import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import koreanize_matplotlib

df = pd.read_csv('Titanic.csv')

# 1. **Titanic 데이터셋 이해**
#    - seaborn 라이브러리로 titanic 데이터 로드: `sns.load_dataset('titanic')`
#    - 데이터 형태: 891행, 15개 열
#    - 주요 열: PassengerId, Survived, Pclass, Age, Sex, Fare, Embarked 등
#    - 결측값 확인: Age(177개), Cabin(687개), Embarked(2개) 결측
print("Titan 데이터셋 이해")
print(df.shape)
print(df[['PassengerId', 'Survived', 'Pclass', 'Age', 'Sex', 'Fare', 'Embarked']])
print(df.isnull().sum()[['Age', 'Cabin', 'Embarked']])

# 2. **좌석 등급별 생존율 분석**
#    - `groupby('pclass')['survived'].mean()` 사용
#    - 예상 결과: 1등석 > 2등석 > 3등석 생존율
#    - 생존율 차이 해석
print("좌석 등급별 생존율 분석")
print(df.groupby('Pclass')['Survived'].mean())


# 3. **성별-좌석등급별 다중 집계**
#    - `groupby(['sex', 'pclass']).agg()` 사용
#    - 각 그룹별 나이 평균, 요금 최대값, 생존율 계산
#    - 가장 높은 생존율 그룹 찾기
print("성별-좌석등급별 다중 집계")
print(df.groupby(['Sex', 'Pclass']).agg({
    'Age':'mean', 'Fare':'max', 'Survived':'mean'
}).round(2))

max_survived_group =  df.groupby(['Sex', 'Pclass']).agg({
    'Survived':'mean'
}).round(2).idxmax()

max_survived_value = df.groupby(['Sex', 'Pclass']).agg({
    'Survived':'mean'
}).round(2).max()

print(f"가장 생존율이 높은 그룹 | {max_survived_group} : {max_survived_value}")


# # 4. **피벗 테이블 생성**
print("**피벗 테이블 생성**")
# #    - 성별 × 좌석등급 생존율 피벗 테이블
pivot = df.pivot_table(
    values='Survived',      
    index='Pclass',          
    columns='Sex',       
    aggfunc='mean'          
).round(2)

print(pivot)
# #    - 히트맵으로 시각화하여 패턴 확인
plt.figure(figsize=(6,6))

sns.heatmap(
    pivot,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=0, vmax=1
)
plt.title('성별 x 좌석등급 생존율 히트맵')
plt.xlabel('성별')
plt.ylabel('좌석등급', rotation=0)
plt.tight_layout()
plt.show()

# 5. **시각화**

fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1행 3열 subplot

# 좌석등급별 생존율 막대 그래프 ──
pclass_survived = df.groupby('Pclass')['Survived'].mean()

axes[0].bar(pclass_survived.index, pclass_survived.values, color=['gold', 'silver', '#cd7f32'])
axes[0].set_title('좌석등급별 생존율')
axes[0].set_xlabel('좌석 등급')
axes[0].set_ylabel('생존율', rotation=0, labelpad=30)
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(['1등석', '2등석', '3등석'])

# 막대 위에 수치 표시
for i, v in zip(pclass_survived.index, pclass_survived.values):
    axes[0].text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=11)


# 성별-좌석등급 생존율 히트맵 
sns.heatmap(
    pivot,               
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    ax=axes[1],
    vmin=0, vmax=1
)
axes[1].set_title('성별-좌석등급 생존율 히트맵')
axes[1].set_xlabel('좌석 등급')
axes[1].set_ylabel('성별', rotation=0, labelpad=30)


# 연령대별 생존자 수 
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 80],
                         labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60+'])

age_survived = df.groupby('AgeGroup', observed=True)['Survived'].sum()

axes[2].bar(age_survived.index, age_survived.values, color='steelblue')
axes[2].set_title('연령대별 생존자 수')
axes[2].set_xlabel('연령대')
axes[2].set_ylabel('생존자 수', rotation=0, labelpad=40)

# 막대 위에 수치 표시
for i, v in enumerate(age_survived.values):
    axes[2].text(i, v + 0.5, str(v), ha='center', fontsize=10)


plt.tight_layout()
plt.show()