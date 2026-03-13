document.getElementById("healthForm").addEventListener("submit", function(e){

    e.preventDefault()

    let formData = new FormData(this)

    fetch("/predict",{
        method:"POST",
        body:formData
    })

    .then(response => response.json())

    .then(data => {

        document.getElementById("result").innerHTML =
        "Heart Disease Risk: " + data.risk + "%"

    })

})