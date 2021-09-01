# code 버전 정리
## 2021
### ver 1 (08.10 initial updated) 
#### ver 1.2 (08.18 initial updated)
08.18
- work14_8_1 업데이트. rotation 코딩 마무리 및 저장까지 전부 완료.
- work13_5 업데이트. score계산 코드 단순화 완료.
##### ver 1.2.1 (08.19 initial updated)
08.19
- 파일 이름을 버전 값으로 변경.
- work14_increasing_data_1_2_1
  - 함수 및 Augmentaion을 수정버전을 적용하기 위해 as fmy, as Aug로 불러오는 방식으로 수정.
  - rot 및 trans 에서 전체에 같은 범위가 적용되던걸 target, bone, skin별 따로 적용되게 수정.
- Augmentation_1_2_1
  - translate에서 데이터가 변경되지 않는 오류 수정 완료.
  - rotation 에서 target, skin, bone 별 따로 rot 되도록 수정 완료.
- functions_my_1_2_1
  - make_trans_offset, make_rot_matrices, abs_vector, calculate_point_to_line_length, data_list_title 추가
- work13_calculate_score_1_2_1
  - 함수 및 Augmentation 을 수정버전을 적용하기 위해 as fmy, as Aug로 불러오는 방식으로 수정.

08.23
- multi target 생성 추가중

#### ver 1.3 (08.25 initial updated)
08.25
- multi target 생성 추가중
- work14_increasing_data_1_3 
  - 함수, 변수, 코드 순서로 코드 구조 변경
  - Augmentation_1_3 에 따라 num_of_data, num_of_data_points 관련 코드 전부 삭제
- Augmentation_1_3  
  - 함수 반복 실행시 기록이 되도록 status 에 += 으로 기록 계속 추가식으로 변경
  - num_of_data, num_of_data_point 인스턴스 변수 삭제

08.27
- functions_my_1_3
  - 공통 함수 외에 전부 삭제
- work14_increasing_data_1_3 
  - 전체 수정 완료.
- Augmentation_1_3 
  - 전체 수정 완료.

08.31
- augmentation 과정을 stack 처리하는 ver 1.3.1 변경과정에서 문제발생해서 전부 리셋
- 모든 파일 변경 사항 없음
- 
##### ver 1.3.1 (09.01 initial updated)
09.01
- work14_increasing_1_3_1
  - 전역변수를 전부 대문자로 변경
  - all_data 리스트를 초기에 만들어서 진행. []의 경우 업데이트가 상호적용되는 것을 확인 가능하였음
- Augmentation_1_3_1
  - self.array_size 삭제 및 자체 계산 방식으로 변경