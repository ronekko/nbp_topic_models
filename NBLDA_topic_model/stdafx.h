// stdafx.h : �W���̃V�X�e�� �C���N���[�h �t�@�C���̃C���N���[�h �t�@�C���A�܂���
// �Q�Ɖ񐔂������A�����܂�ύX����Ȃ��A�v���W�F�N�g��p�̃C���N���[�h �t�@�C��
// ���L�q���܂��B
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>



// TODO: �v���O�����ɕK�v�Ȓǉ��w�b�_�[�������ŎQ�Ƃ��Ă��������B
#ifdef _DEBUG
	#pragma comment(lib, "opencv_core231d.lib")
	#pragma comment(lib, "opencv_imgproc231d.lib")
	#pragma comment(lib, "opencv_highgui231d.lib")
	#pragma comment(lib, "opencv_features2d231d.lib")
#else
	#pragma comment(lib, "opencv_core231.lib")
	#pragma comment(lib, "opencv_imgproc231.lib")
	#pragma comment(lib, "opencv_highgui231.lib")
	#pragma comment(lib, "opencv_features2d231.lib")
#endif


#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <memory>
#include <typeinfo>

#include "direct.h"

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/numeric.hpp>
#include <boost/timer.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/random.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/program_options.hpp>
#include <boost/chrono.hpp>

#include <opencv2/opencv.hpp>