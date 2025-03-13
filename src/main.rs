use quick_xml::reader::Reader;
use serde::{Deserialize, Serialize};
use xml_schema_generator::{Options, into_struct};

fn main() {
    // create struct from XML
    let xml_content = include_str!("test_data/annotation_example.xml");
    let mut reader = Reader::from_str(xml_content);

    if let Ok(root) = into_struct(&mut reader) {
        let struct_as_string = root.to_serde_struct(&Options::quick_xml_de());
        println!("{}", struct_as_string); // this prints the struct Library and Book as listed above
    }

    // parse XML into generated struct
    // let library: Library = quick_xml::de::from_str(xml_content).unwrap();
    // assert_eq!(3, library.book.len());
}
