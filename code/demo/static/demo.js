$(function () { 

$("form").submit(function( event ) {
    showSentence('<span class="neutral">You</span> <span class="neutral">are</span> <span class="neutral">not</span> <span class="polite">very</span> <span class="neutral">helpful</span><span class="neutral">,</span> <span class="neutral">what</span> <span class="neutral">the</span> <span class="impolite">fuck</span> <span class="neutral">are</span> <span class="neutral">you</span> <span class="neutral">doing</span><span class="neutral">?</span>');

  event.preventDefault();
});
    // 
   });

function showSentence(sentence) {
    $("#results .sentence-result").html(sentence);
    $("#results .sentence-result span").lettering()
var tl = new TimelineMax();
TweenLite.set("#results", {perspective:600, autoAlpha:1});
tl.staggerFrom($("#results"), 0.2, {height:"0px",scaleY:0, overflow:"hidden",ease:Circ.easein}, "+=0");
tl.staggerFrom($("#results .sentence-result span"), 0.2, {opacity:0, scale:0, y:80, rotationX:180, transformOrigin:"50% 50% -50",  ease:Back.easeOut}, 0.005, "+=0");

}