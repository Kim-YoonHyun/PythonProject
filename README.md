# code 버전 정리
## 2021
### ver 1 (08.10 initial updated) 
#### 1.2 (08.18 initial updated)
08.18
- work14_8_1 업데이트. rotation 코딩 마무리 및 저장까지 전부 완료.
- work13_5 업데이트. score계산 코드 단순화 완료.
##### 1.2.1 (08.19 initial updated)
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