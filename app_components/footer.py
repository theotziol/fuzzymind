def footer_markdown():
    footer = """
    <style>
    a:link , a:visited{
    color: blue;
    background-color: transparent;
    text-decoration: underline;
    margin-top: 10px;
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
    <p><a style='display: block; text-align: center;' href="https://acta.energy.uth.gr/">ACTA Lab</a></p>
    
    </div>
    """
    return footer
