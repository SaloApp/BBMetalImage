//
//  BBMetalMultipleCamera.swift
//  BBMetalImage
//
//  Created by Kaibo Lu on 10/5/21.
//  Copyright © 2021 Kaibo Lu. All rights reserved.
//

import UIKit
import AVFoundation

@available(iOS 13.0, *)
public class BBMetalMultipleCamera {

    public static var isMultiCamSupported: Bool { AVCaptureMultiCamSession.isMultiCamSupported }
    
    private let session: AVCaptureMultiCamSession
    
    public let backCamera: BBMetalCamera
    public let frontCamera: BBMetalCamera
    
    public init() throws {
        session = .init()
        
        self.backCamera = try BBMetalCamera(captureSession: session, position: .back)
        self.frontCamera = try BBMetalCamera(captureSession: session, position: .front)
    }

}
