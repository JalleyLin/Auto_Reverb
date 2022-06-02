import librosa
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from numpy.fft import fft,rfft
from numpy import cumsum
import pyloudnorm as pyln
import os
import soundfile as sf
import sklearn
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
import os
import beyond.Reaper
import time


base_path = 'C:/reaper_auto_reverb/BeyondReaper/'

def show_feature(masks,feature_list):
	filter_feature = []
	for i,mask in enumerate(masks):
		if mask ==True:
			filter_feature.append(feature_list[i])
	return filter_feature

feature_list = 	['a_rms_avg', 'a_rms_std', 'a_zcr_avg', 'a_zcr_std', 'a_spec_avg', 'a_spec_std', 'a_spf_avg', 'a_spf_std', 'a_spes_avg', 'a_spes_std', 
				'a_spsk_avg', 'a_spsk_std', 'a_spk_avg', 'a_spk_std', 'a_sfx_avg', 'a_sfx_std', 'a_sel_avg', 'a_sel_std', 'a_selm_avg', 'a_selm_std', 
				'a_sem_avg', 'a_sem_std', 'a_semh_avg', 'a_semh_std', 'a_seh_avg', 'a_seh_std', 'a_sea_avg', 'a_a_sea_std', 'a_spen_avg', 'a_spen_std', 
				'a_mfcc1_avg', 'a_mfcc1_std', 'a_mfcc2_avg', 'a_mfcc2_std', 'a_mfcc3_avg', 'a_mfcc3_std', 'a_mfcc4_avg', 'a_mfcc4_std', 'a_mfcc5_avg', 
				'a_mfcc5_std', 'a_mfcc6_avg', 'a_mfcc6_std', 'a_mfcc7_avg', 'a_mfcc7_std', 'a_mfcc8_avg', 'a_mfcc8_std', 'a_mfcc9_avg', 'a_mfcc9_std', 
				'a_mfcc10_avg', 'a_mfcc10_std', 'a_mfcc11_avg', 'a_mfcc11_std', 'a_mfcc12_avg', 'a_mfcc12_std', 'a_mfcc13_avg', 'a_mfcc13_std', 'a_dmfcc1_avg', 
				'a_dmfcc1_std', 'a_dmfcc2_avg', 'a_dmfcc2_std', 'a_dmfcc3_avg', 'a_dmfcc3_std', 'a_dmfcc4_avg', 'a_dmfcc4_std', 'a_dmfcc5_avg', 'a_dmfcc5_std', 
				'a_dmfcc6_avg', 'a_dmfcc6_std', 'a_dmfcc7_avg', 'a_dmfcc7_std', 'a_dmfcc8_avg', 'a_dmfcc8_std', 'a_dmfcc9_avg', 'a_dmfcc9_std', 'a_dmfcc10_avg', 
				'a_dmfcc10_std', 'a_dmfcc11_avg', 'a_dmfcc11_std', 'a_dmfcc12_avg', 'a_dmfcc12_std', 'a_dmfcc13_avg', 'a_dmfcc13_std', 'a_bpm', 'v_rms_avg', 
				'v_rms_std', 'v_zcr_avg', 'v_zcr_std', 'v_spec_avg', 'v_spec_std', 'v_spf_avg', 'v_spf_std', 'v_spes_avg', 'v_spes_std', 'v_spsk_avg', 
				'v_spsk_std', 'v_spk_avg', 'v_spk_std', 'v_sfx_avg', 'v_sfx_std', 'v_sel_avg', 'v_sel_std', 'v_selm_avg', 'v_selm_std', 'v_sem_avg', 
				'v_sem_std', 'v_semh_avg', 'v_semh_std', 'v_seh_avg', 'v_seh_std', 'v_sea_avg', 'v_a_sea_std', 'v_spen_avg', 'v_spen_std', 'v_mfcc1_avg', 
				'v_mfcc1_std', 'v_mfcc2_avg', 'v_mfcc2_std', 'v_mfcc3_avg', 'v_mfcc3_std', 'v_mfcc4_avg', 'v_mfcc4_std', 'v_mfcc5_avg', 'v_mfcc5_std', 
				'v_mfcc6_avg', 'v_mfcc6_std', 'v_mfcc7_avg', 'v_mfcc7_std', 'v_mfcc8_avg', 'v_mfcc8_std', 'v_mfcc9_avg', 'v_mfcc9_std', 'v_mfcc10_avg', 
				'v_mfcc10_std', 'v_mfcc11_avg', 'v_mfcc11_std', 'v_mfcc12_avg', 'v_mfcc12_std', 'v_mfcc13_avg', 'v_mfcc13_std', 'v_dmfcc1_avg', 'v_dmfcc1_std', 
				'v_dmfcc2_avg', 'v_dmfcc2_std', 'v_dmfcc3_avg', 'v_dmfcc3_std', 'v_dmfcc4_avg', 'v_dmfcc4_std', 'v_dmfcc5_avg', 'v_dmfcc5_std', 'v_dmfcc6_avg', 
				'v_dmfcc6_std', 'v_dmfcc7_avg', 'v_dmfcc7_std', 'v_dmfcc8_avg', 'v_dmfcc8_std', 'v_dmfcc9_avg', 'v_dmfcc9_std', 'v_dmfcc10_avg', 
				'v_dmfcc10_std', 'v_dmfcc11_avg', 'v_dmfcc11_std', 'v_dmfcc12_avg', 'v_dmfcc12_std', 'v_dmfcc13_avg', 'v_dmfcc13_std', 'v_bpm', 
				'a_brightness', 'a_lowenergy', 'a_dissonance_dmean', 'a_dissonance_dmean2', 'a_dissonance_dvar', 'a_dissonance_dvar2', 'a_dissonance_max', 
				'a_dissonance_mean', 'a_dissonance_median', 'a_dissonance_min', 'a_dissonance_var', 'a_pulseclarity', 'a_density', 'a_duration_mean', 
				'a_duration_std', 'a_duration_median', 'a_bpm_histogram_first_peak_bpm_dmean', 'a_bpm_histogram_first_peak_bpm_dmean2', 
				'a_bpm_histogram_first_peak_bpm_dvar', 'a_bpm_histogram_first_peak_bpm_dvar2', 'a_bpm_histogram_first_peak_bpm_max', 
				'a_bpm_histogram_first_peak_bpm_mean', 'a_bpm_histogram_first_peak_bpm_median', 'a_bpm_histogram_first_peak_bpm_min', 
				'a_bpm_histogram_first_peak_bpm_var', 'a_bpm_histogram_first_peak_spread_dmean', 'a_bpm_histogram_first_peak_spread_dmean2', 
				'a_bpm_histogram_first_peak_spread_dvar', 'a_bpm_histogram_first_peak_spread_dvar2', 'a_bpm_histogram_first_peak_spread_max', 
				'a_bpm_histogram_first_peak_spread_mean', 'a_bpm_histogram_first_peak_spread_median', 'a_bpm_histogram_first_peak_spread_min', 
				'a_bpm_histogram_first_peak_spread_var', 'a_bpm_histogram_first_peak_weight_dmean', 'a_bpm_histogram_first_peak_weight_dmean2', 
				'a_bpm_histogram_first_peak_weight_dvar', 'a_bpm_histogram_first_peak_weight_dvar2', 'a_bpm_histogram_first_peak_weight_max', 
				'a_bpm_histogram_first_peak_weight_mean', 'a_bpm_histogram_first_peak_weight_median', 'a_bpm_histogram_first_peak_weight_min', 
				'a_bpm_histogram_first_peak_weight_var', 'a_bpm_histogram_second_peak_bpm_dmean', 'a_bpm_histogram_second_peak_bpm_dmean2', 
				'a_bpm_histogram_second_peak_bpm_dvar', 'a_bpm_histogram_second_peak_bpm_dvar2', 'a_bpm_histogram_second_peak_bpm_max', 
				'a_bpm_histogram_second_peak_bpm_mean', 'a_bpm_histogram_second_peak_bpm_median', 'a_bpm_histogram_second_peak_bpm_min', 
				'a_bpm_histogram_second_peak_bpm_var', 'a_bpm_histogram_second_peak_spread_dmean', 'a_bpm_histogram_second_peak_spread_dmean2', 
				'a_bpm_histogram_second_peak_spread_dvar', 'a_bpm_histogram_second_peak_spread_dvar2', 'a_bpm_histogram_second_peak_spread_max', 
				'a_bpm_histogram_second_peak_spread_mean', 'a_bpm_histogram_second_peak_spread_median', 'a_bpm_histogram_second_peak_spread_min', 
				'a_bpm_histogram_second_peak_spread_var', 'a_bpm_histogram_second_peak_weight_dmean', 'a_bpm_histogram_second_peak_weight_dmean2', 
				'a_bpm_histogram_second_peak_weight_dvar', 'a_bpm_histogram_second_peak_weight_dvar2', 'a_bpm_histogram_second_peak_weight_max', 
				'a_bpm_histogram_second_peak_weight_mean', 'a_bpm_histogram_second_peak_weight_median', 'a_bpm_histogram_second_peak_weight_min', 
				'a_bpm_histogram_second_peak_weight_var', 'v_brightness', 'v_lowenergy', 'v_dissonance_dmean', 'v_dissonance_dmean2', 'v_dissonance_dvar', 
				'v_dissonance_dvar2', 'v_dissonance_max', 'v_dissonance_mean', 'v_dissonance_median', 'v_dissonance_min', 'v_dissonance_var', 
				'v_pulseclarity', 'v_density', 'v_duration_mean', 'v_duration_std', 'v_duration_median', 'v_bpm_histogram_first_peak_bpm_dmean', 
				'v_bpm_histogram_first_peak_bpm_dmean2', 'v_bpm_histogram_first_peak_bpm_dvar', 'v_bpm_histogram_first_peak_bpm_dvar2', 
				'v_bpm_histogram_first_peak_bpm_max', 'v_bpm_histogram_first_peak_bpm_mean', 'v_bpm_histogram_first_peak_bpm_median', 
				'v_bpm_histogram_first_peak_bpm_min', 'v_bpm_histogram_first_peak_bpm_var', 'v_bpm_histogram_first_peak_spread_dmean', 
				'v_bpm_histogram_first_peak_spread_dmean2', 'v_bpm_histogram_first_peak_spread_dvar', 'v_bpm_histogram_first_peak_spread_dvar2', 
				'v_bpm_histogram_first_peak_spread_max', 'v_bpm_histogram_first_peak_spread_mean', 'v_bpm_histogram_first_peak_spread_median', 
				'v_bpm_histogram_first_peak_spread_min', 'v_bpm_histogram_first_peak_spread_var', 'v_bpm_histogram_first_peak_weight_dmean', 
				'v_bpm_histogram_first_peak_weight_dmean2', 'v_bpm_histogram_first_peak_weight_dvar', 'v_bpm_histogram_first_peak_weight_dvar2', 
				'v_bpm_histogram_first_peak_weight_max', 'v_bpm_histogram_first_peak_weight_mean', 'v_bpm_histogram_first_peak_weight_median', 
				'v_bpm_histogram_first_peak_weight_min', 'v_bpm_histogram_first_peak_weight_var', 'v_bpm_histogram_second_peak_bpm_dmean', 
				'v_bpm_histogram_second_peak_bpm_dmean2', 'v_bpm_histogram_second_peak_bpm_dvar', 'v_bpm_histogram_second_peak_bpm_dvar2', 
				'v_bpm_histogram_second_peak_bpm_max', 'v_bpm_histogram_second_peak_bpm_mean', 'v_bpm_histogram_second_peak_bpm_median', 
				'v_bpm_histogram_second_peak_bpm_min', 'v_bpm_histogram_second_peak_bpm_var', 'v_bpm_histogram_second_peak_spread_dmean', 
				'v_bpm_histogram_second_peak_spread_dmean2', 'v_bpm_histogram_second_peak_spread_dvar', 'v_bpm_histogram_second_peak_spread_dvar2', 
				'v_bpm_histogram_second_peak_spread_max', 'v_bpm_histogram_second_peak_spread_mean', 'v_bpm_histogram_second_peak_spread_median', 
				'v_bpm_histogram_second_peak_spread_min', 'v_bpm_histogram_second_peak_spread_var', 'v_bpm_histogram_second_peak_weight_dmean', 
				'v_bpm_histogram_second_peak_weight_dmean2', 'v_bpm_histogram_second_peak_weight_dvar', 'v_bpm_histogram_second_peak_weight_dvar2', 
				'v_bpm_histogram_second_peak_weight_max', 'v_bpm_histogram_second_peak_weight_mean', 'v_bpm_histogram_second_peak_weight_median', 
				'v_bpm_histogram_second_peak_weight_min', 'v_bpm_histogram_second_peak_weight_var']

# new_feature_list = ['a_mfcc5_std','a_dmfcc1_std','v_semh_avg','a_mfcc11_avg','a_dmfcc5_std','a_dmfcc11_std','a_pulseclarity','v_dissonance_dvar',
# 					'a_spec_std','a_spen_std','v_mfcc11_std','v_dmfcc1_std','v_dissonance_var','a_sel_std','a_selm_avg','a_mfcc4_avg','a_mfcc6_std']

#初筛

constant_lsit = [15,98,182,183,184,185,190,191,192,193,194,199,200,201,202,203,208,209,210,
					211,212,217,218,219,220,221,226,227,228,229,230,235,252,253,254,255,
 					260,261,262,263,264,269,270,271,272,273,278,279,280,281,282,287,288,289,
 					290,291,296,297,298,299,300,305]
drop_list = []

for i in constant_lsit:
	drop_list.append(feature_list[i])

new_feature_list = []

for feature in feature_list:
	if feature not in drop_list:
		new_feature_list.append(feature)


print(new_feature_list)

#划分训练集
data = pd.read_csv("final_normalized_new(classifier3).csv")
X = data[new_feature_list]
y = data[['reverb']]



X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2,random_state=6)

print(X_test.shape)




#特征筛选

select = SelectPercentile(percentile = 20)

select.fit(X_train, y_train)

X_train_select = select.transform(X_train)

X_test_select = select.transform(X_test)

print(X_train_select.shape)

mask = select.get_support()

mask = show_feature(mask,feature_list)

print(mask)

#特征筛选2.0

# select = SelectFromModel(RandomForestClassifier(n_estimators=1000,random_state=42),threshold = '1.7*mean') 

# select.fit(X_train, y_train)

# X_train_select = select.transform(X_train)

# X_test_select = select.transform(X_test)

# print(X_train_select.shape)

# mask = select.get_support()

# mask = show_feature(mask,feature_list)

# print(mask)




# 测试模型

# dummy_clf = DummyClassifier(strategy="most_frequent")

# dummy_clf.fit(X_train_select,y_train.values.ravel())

# print('dummy_training_score:{}'.format(dummy_clf.score(X_train_select,y_train)))

# print('dummy_test_score:{}'.format(dummy_clf.score(X_test_select,y_test)))


#training 训练模型（贝叶斯）

# reg = GaussianNB()

# reg.fit(X_train_select,y_train.values.ravel())



#training 训练模型（随机森林）

reg = RandomForestClassifier(n_estimators=1000)

reg.fit(X_train_select,y_train.values.ravel())



# training 训练模型（线性回归）

# reg = linear_model.LinearRegression()

# reg.fit(X_train_select,y_train.values.ravel())



#training 训练模型（SVM）
# reg = SVC(C=1,gamma=0.5,kernel='rbf')

# reg.fit(X_train_select,y_train.values.ravel())



# 得到特征重要性
# importances = list(reg.feature_importances_)  
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)] 
# print(type(feature_importances))
# # 排序-降序，key参数决定按哪一列进行排序，lambda函数确定出按第二列排序
# feature_importances = sorted(feature_importances, key =lambda x:x[1], reverse = True)
# print(feature_importances)
# # 对应进行打印
# [print('Variable: {}         Importance: {}'.format(*pair)) for pair in feature_importances]





# # print(reg.intercept_)
print('training_score:{}'.format(reg.score(X_train_select,y_train)))
print('test_score:{}'.format(reg.score(X_test_select,y_test)))


# #交叉验证

# # regs = SVC(kernel='rbf')

# pipe = Pipeline([('select', SelectPercentile(percentile = 20)),('rf', RandomForestClassifier(n_estimators=1000))])

# # regs = RandomForestClassifier(n_estimators=1000)


# sfolder = StratifiedKFold(n_splits=5,random_state=6,shuffle=True)

# cross_result = cross_val_score(pipe,X,np.array(y).ravel(),cv = sfolder)

# print(cross_result)
# print(np.mean(cross_result))




#网格搜索

# best_score = 0
# Cs = [0.001,0.6,0.8,1,1.2,1.4,2]
# sfolder = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)

# for i,gamma in enumerate([0.2,0.3,0.4,0.5,0.6,0.8,1]):
# 	C = Cs[i]
# 	svm = SVC(kernel='rbf',gamma=gamma,C=C)
# 	score = np.mean(cross_val_score(svm,X,np.array(y).ravel(),cv = sfolder))
# 	if score > best_score:
# 		best_score = score
# 		bset_param = {'C':C, 'gamma':gamma}

# print(best_score)
# print(bset_param)

#测试用的数据名单[更换测试用例，请修改item参数]
item = 4
list_for_testing = ['叶锦如 光 part1', '和邦格力 海阔天空 part1', '唐爽 想你的365天', '朱清琳 虫儿飞', '颜丙沂 泡沫']

#自动混音工程的生成模块
new_data = pd.read_csv('final_reverb_output.csv')
new_X = new_data[new_feature_list]
new_X_Select = select.transform(new_X)

new_X_Select = np.array(new_X_Select)
new_vocal_filename = new_data['filename1']
new_ac_filename = new_data['filename2']

cout = 0
for filename in new_vocal_filename:
	if list_for_testing[item] in filename: #搜索目标文件
		print(cout)
		print(new_vocal_filename[cout])
		print(new_ac_filename[cout])

		print(new_X_Select[cout])
		predict_result = reg.predict([new_X_Select[cout]])
		reverb_type = predict_result[0]

		os.system('start D:/Reaper-Win-Portable/reaper.exe') #start是为了让os进程不会中止，可以继续
		time.sleep(5)
		Reaper.Main_openProject(base_path+'reverb_module.rpp')
		time.sleep(5)

		#insert track1
		reaper_media_track = Reaper.GetTrack(0, 0) #第二个参数，0是第一个，1是第二个
		Reaper.SetOnlyTrackSelected(reaper_media_track) #选中
		Reaper.InsertMedia(os.path.join('C:/reaper_auto_reverb/BeyondReaper/vocal for test', new_vocal_filename[cout]), 0)
		

		#insert reverb fx
		if reverb_type == 0:
			fxchain_file =  'short_verb' + '.RfxChain'
		elif reverb_type == 1:
			fxchain_file =  'mid_verb' + '.RfxChain'
		else:
			fxchain_file =  'long_verb' + '.RfxChain'
		print(fxchain_file)
		Reaper.TrackFX_AddByName(reaper_media_track, fxchain_file, 0, -1)
		time.sleep(2)

		#insert track2
		Reaper.CSurf_GoStart()  #转移光标位置
		clip_start_time_sec = Reaper.TimeMap2_QNToTime(0, 0)
		Reaper.MoveEditCursor(clip_start_time_sec, False)
		reaper_media_track = Reaper.GetTrack(0, 1) #第二个参数，0是第一个，1是第二个
		Reaper.SetOnlyTrackSelected(reaper_media_track) #选中
		Reaper.InsertMedia(os.path.join('C:/reaper_auto_reverb/BeyondReaper/ac for test', new_ac_filename[cout]), 0)


		break
	cout += 1





