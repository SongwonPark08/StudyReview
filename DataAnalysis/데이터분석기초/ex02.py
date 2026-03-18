import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

data = {
        'name': ['김철수', '이영희', '박민수', '최지영', '정태현', '한소희', '윤상호', '배수진'],
        'age': [25, 30, 35, 28, 42, 31, 29, 27],
        'city': ['서울', '부산', '대구', '서울', '광주', '서울', '부산', '대구'],
        'salary': [3500, 4200, 3800, 4500, 5200, 3900, 3600, 4100],
        'department': ['개발', '마케팅', '개발', '기획', '개발', '마케팅', '기획', '개발'],
        'experience': [2, 5 , 8, 3, 12, 6, 4, 3]
    }

df = pd.DataFrame(data)

print("1. **DataFrame 생성 및 기본 정보 확인**")
print(f"DataFrame : {df}")
print(f"\nInfo:")
df.info()
print(f"\nDescribe : ")
print(df.describe())
print("-"*50)

print("**단일 조건 필터링**")
# 30세 이상 직원 필터링
th_filter = df['age'] >= 30
print("나이가 30 이상인 직원:")
print(df[th_filter]['name'])

# 서울에 거주하는 직원 필터링
seoul_filter = df['city'] == '서울'
print("\n서울에 거주하는 직원 : ")
print(df[seoul_filter]['name'])

# 급여 4000이상 직원 필터링
ft_filter = df['salary'] >= 4000
print("\n급여가 4000 이상인 직원 : ")
print(df[ft_filter]['name'])

print("**복합 조건 필터링**")
# 서울 거주 급여 4000이상
cond1 = seoul_filter & ft_filter
print("서울 거주 AND 급여 4000 이상 : ")
print(df[cond1]['name'])

cond2 = (df['department'] == '개발') | (df['salary'] >= 4500)
print("\n개발팀 OR 급여 4500 이상 : ")
print(df[cond2]['name'])

cond3 = df['department'] == '개발'
print("\n개발팀이 아닌 직원 : ")
print(df[~cond3])

print("**기본 통계 및 집계**")
print("부서별 평균 급여 : ") 
print(df.groupby(['department'])['salary'].mean())

print("\n도시별 직원 수 : ")
print(df['city'].value_counts())

print("\n경력 통계 분석 : ")
print(df['experience'].describe())

