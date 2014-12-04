$(function () { 
    $("form").submit(function( event ) {
        event.preventDefault();
        $.ajax({
          url: "classify/",
          type: 'post',
          data: $("form").serialize(),
          dataType: 'json',
          success: (function(data) {
            var sentences = data.sentences;
            for (i = 0; i < sentences.length; i++) { 
                showSentence(""+sentences[i].sentence, sentences[i].sentenceClass);
            }
            
            }),
          complete:  (function(data) {
            console.log("Fail");
            }),
        });
          
    });
});
function showSentence(sentence, sentenceClass) {
    if(sentence == "") {
        $("#results .sentence-result").html('<div class="no-input">Do you want to know how I work? Fill in a sentence first!</div>');
        $("#results .sentence-result").lettering('words');
        updateClass("neural");
    }
    else {
        $("#results .sentence-result").html(sentence);
        console.log(sentenceClass);
        updateClass(sentenceClass);
    }
    $("#results .sentence-result span").lettering();
    var tl = new TimelineMax();
    TweenLite.set("#results", {perspective:600, autoAlpha:1});
    tl.staggerFrom($("#results"), 0.2, {scaleY:0, overflow:"hidden",ease:Circ.easein}, "+=0");
    tl.staggerFrom($("#results .sentence-result span"), 0.2, {opacity:0, scale:0, y:80, rotationX:180, transformOrigin:"50% 50% -50",  ease:Back.easeOut}, 0.005, "+=0");

}
function updateClass(newClass) {
    if(newClass == "polite") {
        $("#results .class-label").addClass("label-success");
        $("#results .class-label").removeClass("label-default");
        $("#results .class-label").removeClass("label-danger");
        $("#results .class-label").html("Polite");
    }
    else if(newClass == "impolite") {
        $("#results .class-label").removeClass("label-success");
        $("#results .class-label").removeClass("label-default");
        $("#results .class-label").addClass("label-danger");
        $("#results .class-label").html("Impolite");
    }
    else {
        $("#results .class-label").removeClass("label-success");
        $("#results .class-label").addClass("label-default");
        $("#results .class-label").removeClass("label-danger");
        $("#results .class-label").html("Neutral");
    }
}