from demosite.settings import PROJECT_ROOT, STATICFILES_DIRS
import subprocess
import os
import json
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
from ibm_watson import PersonalityInsightsV3
from ibm_watson import ToneAnalyzerV3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.stem import WordNetLemmatizer
from tinytag import TinyTag

mediaAbsDir = os.path.join(PROJECT_ROOT, os.pardir, 'media')
audPath = os.path.abspath(os.path.join(mediaAbsDir, 'audio.flac'))
splitDir = os.path.abspath(os.path.join(mediaAbsDir, 'splits'))
result = {}
transcriptList = []
allText = ''
splitTime = 0

IBM_SPEECH_TO_TEXT_API_KEY = '{apikey}'
IBM_SPEECH_TO_TEXT_SERVICE_URL = '{url}'
IBM_PERSONALITY_INSIGHTS_API_KEY = '{apikey}'
IBM_PERSONALITY_INSIGHTS_SERVICE_URL = '{url}'
IBM_TONE_ANALYZER_API_KEY = '{apikey}'
IBM_TONE_ANALYZER_SERVICE_URL = '{url}'


def get_audio(file_name):
    # ffmpeg -i "video.mp4" -f flac -sample_fmt s16 -ar 16000 audio-file.flac
    vid_path = os.path.abspath(os.path.join(mediaAbsDir, file_name))
    print(vid_path)
    print(audPath)
    command = ['ffmpeg', '-i', vid_path, '-f', 'flac', '-sample_fmt', 's16', '-ar', '16000', audPath, '-y']
    print(*command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    print('getAudio finished')


def split_audio():
    global splitTime
    # mkdir I:\Atharva\Documents\GitHub\django-upload-example\media\splits
    command = ['mkdir', os.path.abspath(os.path.join(mediaAbsDir, 'splits'))]
    print(*command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)

    tag = TinyTag.get(audPath)
    splitTime = int(tag.duration / 4) + 1

    # ffmpeg -i ./audio-file.flac -f segment -segment_time 30 -c copy ./splits/out%03d.flac
    command = ['ffmpeg', '-i', audPath, '-f', 'segment', '-segment_time', str(splitTime), '-c', 'copy',
               os.path.abspath(os.path.join(splitDir, 'out%03d.flac')), '-y']
    print(*command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    print('splitAudio finished')


def get_transcript_list():
    global result, allText
    authenticator = IAMAuthenticator(IBM_SPEECH_TO_TEXT_API_KEY)
    speech_to_text = SpeechToTextV1(
        authenticator=authenticator
    )
    speech_to_text.set_service_url(IBM_SPEECH_TO_TEXT_SERVICE_URL)
    for entry in sorted(os.listdir(splitDir)):
        with open(os.path.join(splitDir, entry), 'rb') as audio_file:
            result_text = speech_to_text.recognize(
                audio=audio_file,
                content_type='audio/flac',
                model='en-US_BroadbandModel',
                smart_formatting=True
            )
        transcript = ""
        for x in result_text.get_result()['results']:
            transcript += x['alternatives'][0]['transcript']
        transcriptList.append(transcript)
        allText += transcript
    result['transcript'] = transcriptList
    with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json'), 'w') as json_file:
        json.dump(result, json_file)
    print('getTranscriptList finished')


def get_vader_sentiment():
    global result
    sia = SentimentIntensityAnalyzer()
    sentiment_scores_list = []
    for text in transcriptList:
        sentiment_scores_list.append(sia.polarity_scores(text))
    y_axis = {'pos': [x['pos'] for x in sentiment_scores_list], 'neg': [x['neg'] for x in sentiment_scores_list],
              'compound': [x['compound'] for x in sentiment_scores_list]}
    x_axis = list(range(1, 1 + len(y_axis['neg'])))
    result['sentiment'] = {'yAxis': y_axis, 'xAxis': x_axis}
    with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json'), 'w') as json_file:
        json.dump(result, json_file)
    print('getVaderSentiment finished')


def word_frequency_v1():
    global result, allText
    lemmatizer = WordNetLemmatizer()
    word_list = nltk.word_tokenize(allText)
    lema_all_text = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    all_words = nltk.tokenize.word_tokenize(lema_all_text)
    stopwords = nltk.corpus.stopwords.words('english')
    all_word_except_stop_dist = nltk.FreqDist(w.lower() for w in all_words if w.lower() not in stopwords)
    most_common = all_word_except_stop_dist.most_common(10)
    word_frequency_data = [list(dict(most_common).keys()), list(dict(most_common).values())]
    # plt.barh(list(dict(mostCommon).keys()), list(dict(mostCommon).values()))
    # plt.show()
    # print('\n')

    result['wordFrequencyData'] = word_frequency_data
    with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json'), 'w') as json_file:
        json.dump(result, json_file)


def word_frequency():
    global result
    word_frequency_data = []
    lemmatizer = WordNetLemmatizer()
    for x in transcriptList:
        word_list = nltk.word_tokenize(x)
        lema_all_text = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
        all_words = nltk.tokenize.word_tokenize(lema_all_text)
        stopwords = nltk.corpus.stopwords.words('english')
        all_word_except_stop_dist = nltk.FreqDist(w.lower() for w in all_words if w.lower() not in stopwords)
        most_common = all_word_except_stop_dist.most_common(10)
        word_frequency_data.append([list(dict(most_common).keys()), list(dict(most_common).values())])
        # plt.barh(list(dict(mostCommon).keys()), list(dict(mostCommon).values()))
        # plt.show()
        # print('\n')
    result['wordFrequencyData'] = word_frequency_data
    with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json'), 'w') as json_file:
        json.dump(result, json_file)

    list2 = []
    for frqData in result['wordFrequencyData']:
        list1 = []
        for x, y in dict(zip(frqData[0], frqData[1])).items():
            list1.append({'text': x, 'size': y})
        list2.append(list1)

    result['wordcloud'] = list2
    with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json'), 'w') as json_file:
        json.dump(result, json_file)

    print('wordCloud finished')


def get_personality_insights():
    global allText, result
    authenticator = IAMAuthenticator(IBM_PERSONALITY_INSIGHTS_API_KEY)
    personality_insights = PersonalityInsightsV3(
        version='2017-10-13',
        authenticator=authenticator
    )
    personality_insights.set_service_url(IBM_PERSONALITY_INSIGHTS_SERVICE_URL)
    profile = personality_insights.profile(allText, accept='application/json', raw_scores=True).get_result()

    x_axis = []
    per_list = []
    raw_list = []
    for characteristic in profile['personality']:
        # print('Scores for facets in {}'.format(characteristic['name']))
        # print('General score for {}: {}'.format(characteristic['name'], characteristic['percentile']))
        for facet in characteristic['children']:
            x_axis.append(facet['name'])
            per_list.append(facet['percentile'] * 100)
            raw_list.append(facet['raw_score'] * 100)
    result['personality'] = {}
    result['personality']['Personality'] = {'labels': x_axis, 'per': per_list, 'raw': raw_list}
    with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json'), 'w') as json_file:
        json.dump(result, json_file)

    per_list = []
    raw_list = []
    x_axis = []
    # print("Scores for Values")
    for value in profile['values']:
        per_list.append(value['percentile'] * 100)
        raw_list.append(value['raw_score'] * 100)
        x_axis.append(value['name'])
    result['personality']['Values'] = {'labels': x_axis, 'per': per_list, 'raw': raw_list}
    with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json'), 'w') as json_file:
        json.dump(result, json_file)

    per_list = []
    raw_list = []
    x_axis = []
    print("Scores for Needs")
    for need in profile['needs']:
        per_list.append(need['percentile'] * 100)
        raw_list.append(need['raw_score'] * 100)
        x_axis.append(need['name'])
    result['personality']['Needs'] = {'labels': x_axis, 'per': per_list, 'raw': raw_list}
    with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json'), 'w') as json_file:
        json.dump(result, json_file)
    print('personalityInsights finished')


def tone_analysis():
    global result, splitTime, transcriptList
    authenticator = IAMAuthenticator(IBM_TONE_ANALYZER_API_KEY)
    tone_analyzer = ToneAnalyzerV3(
        version='2017-09-21',
        authenticator=authenticator
    )

    tone_analyzer.set_service_url(IBM_TONE_ANALYZER_SERVICE_URL)
    tone_list = []
    for x in transcriptList:
        tone_list.append(tone_analyzer.tone({'text': x}, content_type='application/json'
                                            ).get_result())
    time_list = []
    tone_name_list = []
    i = 1
    fin_tones = ''
    for tone in tone_list:
        if len(tone['document_tone']['tones']) > 0:
            for toneType in tone['document_tone']['tones']:
                time_list.append([splitTime * (i - 1), splitTime * i])
                tone_name_list.append(toneType['tone_name'])
                fin_tones += 'Tone: {} was detected from: {} sec to {} sec\n'.format(toneType['tone_name'],
                                                                                     splitTime * (i - 1),
                                                                                     splitTime * i)
        i += 1

    result['tones'] = {'time': time_list, 'tone': tone_name_list}
    result['finTones'] = fin_tones
    with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json'), 'w') as json_file:
        json.dump(result, json_file)
    print('toneAnalysis finished')


def start_analysis(file_name):
    global allText, result, transcriptList, splitTime
    result = {}
    allText = ''
    transcriptList = []
    splitTime = 0
    with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json'), 'w') as json_file:
        json.dump(result, json_file)
    print(file_name)
    get_audio(file_name)
    split_audio()
    get_transcript_list()
    get_vader_sentiment()
    word_frequency_v1()
    get_personality_insights()
    tone_analysis()
    result['allText'] = allText
    with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json'), 'w') as json_file:
        json.dump(result, json_file)
    print('Analysis finished')
