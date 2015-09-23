# Data preparation

This page describes the sources and preparation required to obtain the
dataset we visualize in the paper. We used STATA to process these data so all
the code shown here is written in that language. 

The eventual analysis relies heavily on
a dataset from the Integrated
Public Use Microdata Series (IPUMS, link [ipums.org](http://ipums.org)). The access is free
after registration. Reproduction of these results require the decadal
census data, in particular the variables: 


<table class=MsoNormalTable border=0 cellpadding=0 style='mso-cellspacing:1.5pt;
 mso-yfti-tbllook:1184'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL><a href="https://usa.ipums.org/usa-action/faq#ques33"><b><span
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-bidi-font-family:"Times New Roman";color:#39c;
  mso-fareast-language:NL'>Type</span></b></a></span><b><span lang=NL
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-bidi-font-family:"Times New Roman";mso-fareast-language:
  NL'><o:p></o:p></span></b></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span class=SpellE><b><span lang=NL style='font-size:12.0pt;
  font-family:"Times New Roman";mso-fareast-font-family:"Times New Roman";
  mso-bidi-font-family:"Times New Roman";mso-fareast-language:NL'>Variable</span></b></span><b><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-bidi-font-family:"Times New Roman";mso-fareast-language:
  NL'><o:p></o:p></span></b></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><b><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Label<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/YEAR"><span
  style='color:#39c'>YEAR</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Census <span class=SpellE>year</span><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/DATANUM"><span
  style='color:#39c'>DATANUM</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Data set <span class=SpellE>number</span><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/SERIAL"><span
  style='color:#39c'>SERIAL</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Household <span class=SpellE>serial</span> <span
  class=SpellE>number</span><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/HHWT"><span
  style='color:#39c'>HHWT</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Household <span class=SpellE>weight</span><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/STATEFIP"><span
  style='color:#39c'>STATEFIP</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>State (FIPS code)<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/COUNTY"><span
  style='color:#39c'>COUNTY</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span class=SpellE><span lang=NL style='font-size:12.0pt;font-family:
  "Times New Roman";mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:
  "Times New Roman";mso-fareast-language:NL'>County</span></span><span lang=NL
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-bidi-font-family:"Times New Roman";mso-fareast-language:
  NL'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/METAREA"><span
  style='color:#39c'>METAREA</span></a> (<span class=SpellE>general</span>) <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span class=SpellE><span lang=NL style='font-size:12.0pt;font-family:
  "Times New Roman";mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:
  "Times New Roman";mso-fareast-language:NL'>Metropolitan</span></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-bidi-font-family:"Times New Roman";mso-fareast-language:
  NL'> area [<span class=SpellE>general</span> <span class=SpellE>version</span>]
  <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:8'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/METAREA"><span
  style='color:#39c'>METAREAD</span></a> (<span class=SpellE>detailed</span>) <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span class=SpellE><span lang=NL style='font-size:12.0pt;font-family:
  "Times New Roman";mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:
  "Times New Roman";mso-fareast-language:NL'>Metropolitan</span></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-bidi-font-family:"Times New Roman";mso-fareast-language:
  NL'> area [<span class=SpellE>detailed</span> <span class=SpellE>version</span>]
  <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:9'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/GQ"><span style='color:#39c'>GQ</span></a>
  <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Group <span class=SpellE>quarters</span> status<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:10'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/VALUEH"><span
  style='color:#39c'>VALUEH</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>House <span class=SpellE>value</span><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:11'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/ROOMS"><span
  style='color:#39c'>ROOMS</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span class=SpellE><span lang=NL style='font-size:12.0pt;font-family:
  "Times New Roman";mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:
  "Times New Roman";mso-fareast-language:NL'>Number</span></span><span lang=NL
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-bidi-font-family:"Times New Roman";mso-fareast-language:
  NL'> of rooms<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:12'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/BUILTYR"><span
  style='color:#39c'>BUILTYR</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Age of <span class=SpellE>structure</span><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:13'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/BUILTYR2"><span
  style='color:#39c'>BUILTYR2</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Age of <span class=SpellE>structure</span>, decade<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:14'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>H<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/UNITSSTR"><span
  style='color:#39c'>UNITSSTR</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Units in <span class=SpellE>structure</span><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:15'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>P<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/PERNUM"><span
  style='color:#39c'>PERNUM</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-ansi-language:EN-US;mso-fareast-language:NL'>Person number in sample unit<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:16'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>P<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/PERWT"><span
  style='color:#39c'>PERWT</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Person <span class=SpellE>weight</span><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:17'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>P<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/SEX"><span style='color:
  #39c'>SEX</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span class=SpellE><span lang=NL style='font-size:12.0pt;font-family:
  "Times New Roman";mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:
  "Times New Roman";mso-fareast-language:NL'>Sex</span></span><span lang=NL
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-bidi-font-family:"Times New Roman";mso-fareast-language:
  NL'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:18'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>P<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/AGE"><span style='color:
  #39c'>AGE</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Age<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:19'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>P<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/RACE"><span
  style='color:#39c'>RACE</span></a> (<span class=SpellE>general</span>) <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Race [<span class=SpellE>general</span> <span
  class=SpellE>version</span>] <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:20'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>P<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/RACE"><span
  style='color:#39c'>RACED</span></a> (<span class=SpellE>detailed</span>) <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>Race [<span class=SpellE>detailed</span> <span
  class=SpellE>version</span>] <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:21'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>P<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/EDUC"><span
  style='color:#39c'>EDUC</span></a> (<span class=SpellE>general</span>) <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span class=SpellE><span lang=NL style='font-size:12.0pt;font-family:
  "Times New Roman";mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:
  "Times New Roman";mso-fareast-language:NL'>Educational</span></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-bidi-font-family:"Times New Roman";mso-fareast-language:
  NL'> <span class=SpellE>attainment</span> [<span class=SpellE>general</span> <span
  class=SpellE>version</span>] <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:22'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>P<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/EDUC"><span
  style='color:#39c'>EDUCD</span></a> (<span class=SpellE>detailed</span>) <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span class=SpellE><span lang=NL style='font-size:12.0pt;font-family:
  "Times New Roman";mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:
  "Times New Roman";mso-fareast-language:NL'>Educational</span></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-bidi-font-family:"Times New Roman";mso-fareast-language:
  NL'> <span class=SpellE>attainment</span> [<span class=SpellE>detailed</span>
  <span class=SpellE>version</span>] <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:23'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>P<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/IND"><span style='color:
  #39c'>IND</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span class=SpellE><span lang=NL style='font-size:12.0pt;font-family:
  "Times New Roman";mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:
  "Times New Roman";mso-fareast-language:NL'>Industry</span></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-bidi-font-family:"Times New Roman";mso-fareast-language:
  NL'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:24;mso-yfti-lastrow:yes'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'>P<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:"Times New Roman";
  mso-fareast-language:NL'><a
  href="https://usa.ipums.org/usa-action/variables/INCWAGE"><span
  style='color:#39c'>INCWAGE</span></a> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span class=SpellE><span lang=NL style='font-size:12.0pt;font-family:
  "Times New Roman";mso-fareast-font-family:"Times New Roman";mso-bidi-font-family:
  "Times New Roman";mso-fareast-language:NL'>Wage</span></span><span lang=NL
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-bidi-font-family:"Times New Roman";mso-fareast-language:
  NL'> <span class=SpellE>and</span> <span class=SpellE>salary</span> <span
  class=SpellE>income</span><o:p></o:p></span></p>
  </td>
 </tr>
</table>

Out of the samples:

<table class=MsoNormalTable border=0 cellpadding=0 style='mso-cellspacing:1.5pt;
 mso-yfti-tbllook:1184'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL><a
  href="https://usa.ipums.org/usa/sampdesc.shtml#us1940a"><span
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'>1940 1%</span></a></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-fareast-language:NL'>1.0%<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'></td>
 </tr>
 <tr style='mso-yfti-irow:1'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL><a
  href="https://usa.ipums.org/usa/sampdesc.shtml#us1950a"><span
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'>1950 1%</span></a></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-fareast-language:NL'>1.0%<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'></td>
 </tr>
 <tr style='mso-yfti-irow:2'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL><a
  href="https://usa.ipums.org/usa/sampdesc.shtml#us1960a"><span
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'>1960 1%</span></a></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-fareast-language:NL'>1.0%<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'></td>
 </tr>
 <tr style='mso-yfti-irow:3'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL><a
  href="https://usa.ipums.org/usa/sampdesc.shtml#us1970c"><span
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'>1970 1% metro fm1</span></a></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-fareast-language:NL'>1.0%<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'></td>
 </tr>
 <tr style='mso-yfti-irow:4'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL><a
  href="https://usa.ipums.org/usa/sampdesc.shtml#us1970d"><span
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'>1970 1% metro fm2</span></a></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-fareast-language:NL'>1.0%<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'></td>
 </tr>
 <tr style='mso-yfti-irow:5'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL><a
  href="https://usa.ipums.org/usa/sampdesc.shtml#us1980a"><span
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'>1980 5% state</span></a></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-fareast-language:NL'>5.0%<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'></td>
 </tr>
 <tr style='mso-yfti-irow:6'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL><a
  href="https://usa.ipums.org/usa/sampdesc.shtml#us1990a"><span
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'>1990 5%</span></a></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-fareast-language:NL'>5.0%<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'></td>
 </tr>
 <tr style='mso-yfti-irow:7'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL><a
  href="https://usa.ipums.org/usa/sampdesc.shtml#us2000a"><span
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'>2000 5%</span></a></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-fareast-language:NL'>5.0%<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'></td>
 </tr>
 <tr style='mso-yfti-irow:8;mso-yfti-lastrow:yes'>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL><a
  href="https://usa.ipums.org/usa/sampdesc.shtml#us2010a"><span
  style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'>2010 ACS</span></a></span><span
  lang=NL style='font-size:12.0pt;font-family:"Times New Roman";mso-fareast-font-family:
  "Times New Roman";mso-fareast-language:NL'> <o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span lang=NL style='font-size:12.0pt;font-family:"Times New Roman";
  mso-fareast-font-family:"Times New Roman";mso-fareast-language:NL'>1.0%<o:p></o:p></span></p>
  </td>
  <td style='padding:.75pt .75pt .75pt .75pt'></td>
 </tr>
</table>

This yields a dataset of around 4Gb. Some of the variables were used to run
hedonic land price and Mincer wage regression, so you might drop variables like
rooms, structure age, sex, age and race, but not industries. 

Running these files requires modifying the bits `[folder]` to the locations the
dataset is loaded from and results are saved to. To construct the relevant variables,
we take the log of average wages and rents. 

```
use "[folder]\microdata.dta", clear
keep incwage valueh perwt year metarea
replace incwage=. if incwage==999999
replace incwage=.  if incwage==0
replace valueh=. if valueh==9999999
replace valueh=.  if valueh==0
egen yearweightwage=total(perwt) if incwage!=., by(year metarea)
gen wage=perwt/yearweightwage*incwage
egen yearweightrent=total(perwt) if valueh!=., by(year metarea)
gen rent=perwt/yearweightrent*valueh
collapse (sum) wage rent, by(year metarea)
gen lwage=log(wage)
gen lrent=log(rent)
keep year metarea lwage lrent
save "[folder]\metarealevel.dta", replace
```

Next, we take the share of total workers that classifies as a manufacturing worker 
or a service worker according to industry digits. We also define highly educated
workers as having a college education or higher:

```
use "[folder]\microdata.dta", clear
keep year metarea statefip county ind educ perwt
drop if metarea==0
egen pop=total(perwt), by(year metarea)
replace pop=log(pop)
egen workers=total(perwt) if ind>1 & ind<994, by(year metarea)
egen workerman=total(perwt) if ind>300& ind<500, by(year metarea)
egen workerserv=total(perwt) if ind>700& ind<900, by(year metarea)
gen manufacturing=workerman/workers
gen service=workerserv/workers
drop workers*
egen obseduc=total(perwt) if educ>0, by(year metarea)
egen higheduc=total(perwt) if educ>6, by(year metarea)
gen college=higheduc/obseduc
```

Finally, we construct the specialization index (the Krugman K index). We preserve
and restore the data to collapse for the specialization indexes and worker sectors
and education:

```
preserve
drop if ind>900 &ind<1000
drop if ind>9000
drop if ind==0
bysort year: egen max=max(ind)
gen ind2=int(ind/10) if max<1000
replace ind2=int(ind/1000) if max>1000
bysort metarea year ind2: egen empl=total(perwt)
collapse (mean) empl, by(metarea year ind2)
bysort year metarea: egen emplmet=total(empl)
bysort year: egen emplnet=total(empl)
bysort year ind2: egen emplind=total(empl)
gen b=abs(empl/emplmet-emplind/emplnat)
bysort metarea year: egen k=total(b)
collapse (mean) k, by(year metarea)
sort metarea year
save "[folder]\specialization.dta", replace
restore 
collapse (mean) pop manufacturing service government college, by(year metarea)
save "[folder]\jobsdata.dta", replace
```

By now, we can put together the data and run the main regression:

```
clear
use "[folder]\metarealevel.dta", clear
drop if metarea==0 | metarea==.
sort metarea year
merge m:1 metarea using weatherdatastation.dta
rm weatherdatastation.dta
drop _merge
merge 1:1 metarea year using "[folder]\jobsdata.dta"
drop _merge
merge 1:1 metarea year using "[folder]\specialization.dta"
foreach var in manufacturing service college k{
    gen we`var'=.
    gen re`var'=.
    foreach x in 1940 1970 1980 1990 2000 2010 { /*1950 has no rents*/
        cap sureg (lwage `var') (lrent `var') if year==`x'
        cap replace we`var'=[lwage]`var'*`var' if year==`x'
        cap replace re`var'=[lrent]`var'*`var' if year==`x' 
    }
}
```

Where `SUREG` refers to the seemingly unrelated regression to allow unobserved
to be correlated across the wage and rent regressions for the same metropolitan
area in a year. This output, in a .csv form, serves as all necessary input for
the visualization exercise. 

```
keep year metarea we* re*
destring metarea, replace
outsheet using "[folder]\robackeffects.csv", comma nolabel replace
```

