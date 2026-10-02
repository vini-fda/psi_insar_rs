//! A minimal HTTP/1.1 server for testing the dataset downloaders without network access.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

/// A canned HTTP response for the mock server.
pub struct MockResponse {
    pub status: u16,
    pub headers: Vec<(&'static str, String)>,
    pub body: Vec<u8>,
}

impl MockResponse {
    /// A JSON response with a `{"message": ...}` body.
    pub fn json(status: u16, message: &str) -> Self {
        Self::json_body(status, format!("{{\"message\":\"{message}\"}}"))
    }

    pub fn json_body(status: u16, body: impl Into<String>) -> Self {
        MockResponse {
            status,
            headers: vec![("Content-Type", "application/json".into())],
            body: body.into().into_bytes(),
        }
    }

    pub fn redirect(location: String) -> Self {
        MockResponse {
            status: 307,
            headers: vec![("Location", location)],
            body: Vec::new(),
        }
    }

    pub fn bytes(content_type: &str, body: Vec<u8>) -> Self {
        MockResponse {
            status: 200,
            headers: vec![("Content-Type", content_type.into())],
            body,
        }
    }
}

/// A request received by the mock server.
#[derive(Debug, Clone)]
pub struct RecordedRequest {
    /// Path and query string.
    pub target: String,
}

impl RecordedRequest {
    /// The target without its query string.
    pub fn path(&self) -> &str {
        self.target.split('?').next().unwrap()
    }
}

/// The requests received so far, shared with the server thread.
#[derive(Clone, Default)]
pub struct MockRequests(Arc<Mutex<Vec<RecordedRequest>>>);

impl MockRequests {
    pub fn all(&self) -> Vec<RecordedRequest> {
        self.0.lock().unwrap().clone()
    }

    pub fn paths(&self) -> Vec<String> {
        self.all().iter().map(|r| r.path().to_string()).collect()
    }

    pub fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

/// Serves `responses` in order, one per connection, and records the requests.
/// Returns the server's base URL (`http://127.0.0.1:<port>`).
pub fn mock_server(responses: Vec<MockResponse>) -> (String, MockRequests) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let requests = MockRequests::default();
    let recorded = requests.clone();
    thread::spawn(move || {
        for response in responses {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let target = request_line.split_whitespace().nth(1).unwrap().to_string();

            let mut content_length = 0;
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap() > 2 {
                if let Some((name, value)) = line.trim_end().split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = value.trim().parse().unwrap();
                }
                line.clear();
            }
            // Read the body, so that closing the socket does not reset the connection.
            let mut body = vec![0; content_length];
            reader.read_exact(&mut body).unwrap();
            recorded.0.lock().unwrap().push(RecordedRequest { target });

            let mut head = format!(
                "HTTP/1.1 {} Mock\r\nContent-Length: {}\r\nConnection: close\r\n",
                response.status,
                response.body.len()
            );
            for (name, value) in &response.headers {
                head.push_str(&format!("{name}: {value}\r\n"));
            }
            head.push_str("\r\n");
            stream.write_all(head.as_bytes()).unwrap();
            stream.write_all(&response.body).unwrap();
        }
    });
    (base, requests)
}
