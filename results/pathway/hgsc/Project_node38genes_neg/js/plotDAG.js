
plotDAG = function(id,json,style) {    
      
     	var thisName = id.split('_')[1];
        
      $('#graphDAG').append("<div id ='" + thisName+"'><div id=\"graphDiv"+ thisName +"\" style=\"height:85%; z-index:3000;\" display=\"none\"></div>\n <div id=\"toolBox\" style='height:15%;' class=\"row\">&nbsp&nbsp <div class = \"column dropup\" display = \"inline\" float =\"left\"> <button class=\"btn btn-primary dropdown-toggle\" type=\"button\" vertical-align=\"top\" data-toggle=\"dropdown\"aria-haspopup=\"true\"> <span class=\"glyphicon glyphicon-download-alt\"></span></button> <ul class= \"dropdown-menu\"> \n<li><a href = \"javascript://\" onclick=\"downloadOntology('pdf', '"+thisName+"', '"+style+"');\">PDF</a></li> \n<li><a href = \"javascript://\" onclick=\"downloadOntology('svg', '"+thisName+"', '"+style+"');\">SVG</a></li> \n<li><a href = \"javascript://\" onclick=\"downloadOntology('png', '"+thisName+"', '"+style+"');\">PNG</a></li> \n</ul> </div> <div class = \"column dropup\"display = \"inline\" float =\"left\"> <button class=\"btn btn-primary dropdown-toggle\" type=\"button\" vertical-align=\"top\" data-toggle=\"dropdown\"aria-haspopup=\"true\"> <span class=\"glyphicon glyphicon-search\"></span></button> <ul class= \"dropdown-menu\" style=\"padding:12px;\"> <button onclick=\"searchViewButton(searchBarValue"+thisName+".value, '"+thisName+"')\" class=\"btn btn-default pull-right\"><i class=\"glyphicon glyphicon-search\"></i></button><input type=\"text\" id=\"searchBarValue"+thisName+"\" class=\"form-control pull-left\" placeholder=\"Search\"> </ul> </div> <div class=\"column\"display = \"inline\" float =\"left\"> <button onclick=\"resize('"+thisName+"')\" class=\"btn btn-primary\"><i class=\"glyphicon glyphicon-resize-full\"></i></button> </div> <div class = \"column\" display= \"inline\" float = \"left\" id = \"legend\"> </div>\n</div> <canvas id=\"hiddenCanvas\"></canvas>");
      
      var data;
      
      data = JSON.parse(json);
      
      console.log("Reports received, accessible as data", data);
      
      var searchTerms = [];
      if (data[0].reports.length != 1) {
         var cancerTypeForSearch = data[0].id.split("_")[1];
         for (var j = 0; j < data[0].reports.length; j++) {
            searchTerms.push(data[0].reports[j]["gID"] + " " + data[0].reports[j]["Name"]); //Hard coded for GO-Term/function of Go-Term
         }
         var JQueryIDTerm = "#searchBarValue" + cancerTypeForSearch;
         $(function() {
            $(JQueryIDTerm).autocomplete({
                source: searchTerms
            });
         });
         var finalIDName = "graphDiv" + data[0].id.split("_")[1]; //hard coded currently
         var params = [];
         params["id"] = data[0].id;
         window.xtrace = new XTraceDAG(document.getElementById(finalIDName), data[0].reports, params);
      }
}