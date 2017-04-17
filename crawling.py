# -*- coding: utf-8 -*-
# Author: onlyzs1023@gmail.com

import scrapy

class PageSpider(scrapy.Spider):
    name = "crawling"
    allowed_domains = ["hc.minhana.net"]
    start_urls = ['https://hc.minhana.net/gallery']

    # Get next_url
    def parse(self, response):
        for i in range(1,1000):
            for next_url in response.css('.imgWrap a::attr(href)').extract():
                next_page = response.urljoin("https:" + next_url)
                yield scrapy.Request(next_page, callback=self.parse_item)
            yield scrapy.Request("https://hc.minhana.net/gallery?p=" + str(i))

    # Get img_url & f_name
    def parse_item(self, response):
        f_name = response.css('#imgDisplay::attr(alt)').extract()
        if len(f_name)!=0 and f_name[0]!="": 
            new_url = response.css('.btnZoom::attr(href)').extract()

            yield {
                'title': f_name[0],
                'urls': new_url[0],
            }
