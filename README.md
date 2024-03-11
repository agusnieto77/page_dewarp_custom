page_dewarp_custom
===========

Corrección de páginas con líneas curvas mediante un modelo de "hoja cúbica": consulte el texto completo en <https://mzucker.github.io/2016/08/15/page-dewarping.html>

Requiere:

 - scipy
 - OpenCV 3.0 o superior
 - Módulo Image de PIL o Pillow
 
Uso:

    page_dewarp_custom.py IMAGE1 [IMAGE2 ...]

Ejemplos:

```bash
    python page_dewarp_custom.py img_in/manifiesto_1900_07_03.jpg
```

```bash    
    python page_dewarp_custom.py img_in/manifiesto_1900_07_03.jpg img_in/manifiesto_1900_07_05.jpg  img_in/manifiesto_1900_07_05_o.jpg img_in/manifiesto_1900_07_16.jpg img_in/manifiesto_1900_07_24.jpg img_in/manifiesto_1900_07_24_o.jpg img_in/manifiesto_1900_07_24_2.jpg img_in/entradas_1900_07_26.jpg img_in/entradas_1900_07_30.jpg
```

```bash   
    find img_in -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec python page_dewarp_custom.py {} +
```    
