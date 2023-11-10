<html>
    <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
    </head>
    <body style="margin: 0 100;background: whitesmoke;">
        <h1>GISModelMaker: </h1>

        <!-- *** Section 0 Basic Information*** --->
        <h2>Section 1: Basic Information</h2>
        <h3>DataSet</h3>
        <p><b>Sample path:</b> {{dataset.sample_path}}</p>
        <p><b>Target path:</b> {{target_path}}</p>
        <p><b>Original Features: </b>{{dataset.feature_names}}</p>
        <p><b>Label:</b> {{dataset.label_name}}</p>
        <h3>Preprocessor</h3>
        <p><b>Method:</b> {{prepro_method}}</p>
        <p><b>Parameter:</b> {{prepro_param}}</p>
        <h3>Model</h3>
        <p><b>Model Name:</b> {{model_name}}</p>
        <p><b>Model Parameter:</b> {{model_param}}</p>

        <!-- *** Section 1 DataSet *** --->
        <h2>Section 1: DataSet</h2>
        <h3> Training Dataset </h3>
        {{ds_stats.get('train')}}
        <h3> Testing Dataset </h3>
        {{ds_stats.get('test')}}
        <h3> Target Dataset </h3>
        {{ds_stats.get('target')}}

        <!-- *** Section 2 Preprocessor *** --->
        <h2>Section 2: Preprocessor</h2>
        <p><b>Number of Feature:</b> {{dataset.X_train.shape[-1]}} -> {{used_feature_number}} </p>
        <h3> Information of Used Preprocessor  </h3>
        {{used_pp_info}}
        <h3> Features Used for Model </h3>
        {{used_features}}

        <!-- *** Section 3 Evaluation *** --->
        <h2>Section 3: Model Evaluation</h2>
        
        <p></p>
        <h3>1. Global Accuracy</h3>
        {{overall_stat.replace('<table border="1" class="dataframe">','<table class="table table-striped">')}}
        <h3>2. Error Matrix</h3>
        {{error_matrix.replace('<table border="1" class="dataframe">','<table class="table table-striped">')}}
        <h3>3. Classification Report (By Scikit-learn)</h3>
        {{clsf_report.replace('<table border="1" class="dataframe">','<table class="table table-striped">')}}

        <h3>4. Confusion Matrix (by Scikit-learn)</h3>
        {{cm_display}}

        <h4>Confusion Matrix Heatmap (Percentage)</h4>
        {{cm_percent_heatmap}}

        <!-- *** Section 3 Generated Map *** --->
        <h2>Section 4: Mapping</h2>
        {{map}}

        <h2>Section 5: Time Statistics</h2>
        <p>Preprocessing Time: {{prepro_time}} s</p>
        <p>Model Training Time: {{train_time}} s</p>
        <p>Model Evaluation Time: {{evaludate_time}} s</p>
        <p>Predicted Map Generalization Time: {{map_time}} s</p>
    </body>
</html>