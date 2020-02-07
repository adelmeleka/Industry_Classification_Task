from flask import Flask, request, render_template
import pickle
import logic

app = Flask(__name__)

@app.route('/job_title/<string:name>', methods=['GET'])
def main_fun(name):
	
	frame_data = logic.read_dataset()

	logic.dataset_cleaning(frame_data)

	X,vectorizer = logic.feature_extractor(frame_data)

	Y,category_dict = logic.label_extractor(frame_data)

	x_train_res, y_train_res,x_test,y_test = logic.split_balance(X,Y)

	# svm_clf = logic.linear_svm(x_train_res, y_train_res)
	#load Saved model
	svm_clf = pickle.load(open('finalized_model.sav', 'rb'))

	result = logic.model_test(name,svm_clf,vectorizer,category_dict)

	return render_template('index.html', title=name,result=result)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
