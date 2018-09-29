/**
 * 更改图片oldURL的类标签
 * @param oldURL: 战舰/zhanjian_01.jpg
 * @param newLabel: 坦克
 */
function updateImageLabel(oldURL, newLabel, token) {
    $.ajaxSetup({
        data: {csrfmiddlewaretoken: token },
    });
    $.ajax({
        url: "/update/label/",
        type: "POST",
        data: {oldURL: oldURL, newLabel: newLabel},
        dataType: "JSON",
        success: function (response) {
            // var data = JSON.parse(response);
            console.log(response.status);
            if (response.status === 0)
            {
                window.location.reload();
            }
        },
        error: function (error) {
            console.error(error);
        }
    });
}

/**
 * 删除图片
 * @param imagePath: 战舰/zhanjian_01.jpg
 */
function deleteImage(imagePath, token) {
    $.ajaxSetup({
        data: {csrfmiddlewaretoken: token }
    });
    console.log(imagePath);
    $.ajax({
        url: "/delete/",
        type: "POST",
        data: {file_path: imagePath},
        dataType: "JSON",
        success: function (response) {
            console.log(response.status);
            if (response.status === 0)
            {
                window.location.reload();
            }
        },
        error: function (error) {
            console.error(error);
        }
    });
}

function startClassify(url, containerID, token) {
    $.ajaxSetup({
        data: {csrfmiddlewaretoken: token }
    });
    $.ajax({
        url: url,
        type: "POST",
        success: function (response) {
            var container = $("#"+containerID);
            container.html(response);
        },
        error: function (error) {
            console.error(error);
        }
    });
}

function startClassify1By1(url, image, containerID, token) {
    console.log(image);
    $.ajaxSetup({
        data: {csrfmiddlewaretoken: token }
    });
    $.ajax({
        url: url,
        type: "POST",
        data: {"image": image},
        success: function (response) {
            var container = $("#"+containerID);
            container.prepend($(response));
        },
        error: function (error) {
            console.error(error);
        }
    });
}


function trainModel(url, params, token) {
    $.ajaxSetup({
        data: {csrfmiddlewaretoken: token }
    });
    $.ajax({
        url: url,
        type: "POST",
        data: params,
        dataType: "JSON",
        success: function (response) {
            console.log(response);
        },
        error: function (error) {
            console.error(error);
        }
    });
}

function startValidate(url, params, token) {
    $.ajaxSetup({
        data: {csrfmiddlewaretoken: token }
    });
    $.ajax({
        url: url,
        type: "POST",
        data: params,
        dataType: "JSON",
        success: function (response) {
            console.log(response);
            $("#" + params["index"]).html(response["index"]);
            $("#" + params["confusion"]).html(response["confusion"]);
            $("#" + params["details"]).html(response["details"]);
        },
        error: function (error) {
            console.error(error);
        }
    });
}







