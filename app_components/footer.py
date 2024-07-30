def footer_markdown():
    footer="""
    <style>
    a:link , a:visited{
    color: blue;
    background-color: transparent;
    text-decoration: underline;
    }
    
    a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
    }
    
    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    }
    </style>
    <div class="footer">
    <p>Developed by <a style='display: block; text-align: center;' href="https://github.com/theotziol" >Theodoros Tziolas</a></p>
    <p>B.Sc. Electrical Engineering, M.Sc. Energy and Automation Systems, Ph.D. in Artificial Intelligence</p>
    
    </div>
    """
    return footer